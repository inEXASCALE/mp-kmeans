import unittest
from unittest.mock import patch

import torch

from mp_kmeans import KMeansPlusPlus


class DistanceDispatchTests(unittest.TestCase):
    def test_mixed_modes_pass_fp32_coordinates_and_threshold_to_euclimix(self):
        X = torch.tensor([[128.015625]])
        centers = torch.tensor([[128.03125]])

        for precision in ('fp16', 'bf16', 'fp32'):
            for correction in ('fp32', 'fp64'):
                kernel = f'{precision}_{correction}'
                with (
                    self.subTest(kernel=kernel),
                    patch('mp_kmeans.clustering.pairwise_squared_distances') as distance,
                ):
                    model = KMeansPlusPlus(n_clusters=1, kernel=kernel, kappa=7.5)
                    self.assertEqual(model._get_dtype(), torch.float32)
                    self.assertIs(model._compute_distances(X, centers), distance.return_value)
                    actual_X, actual_centers = distance.call_args.args
                    torch.testing.assert_close(actual_X, X)
                    torch.testing.assert_close(actual_centers, centers)
                    self.assertEqual(distance.call_args.kwargs, {
                        'precision': precision, 'correction': correction, 'rho': 7.5,
                    })

    def test_pytorch_baselines_return_squared_distances(self):
        X = torch.tensor([[0., 0.], [3., 4.]])
        centers = torch.tensor([[0., 0.], [3., 0.]])
        expected = torch.tensor([[0., 9.], [25., 16.]])

        for kernel, dtype in (('fp32', torch.float32), ('fp64', torch.float64)):
            with self.subTest(kernel=kernel):
                model = KMeansPlusPlus(2, kernel=kernel)
                torch.testing.assert_close(model._compute_distances(X, centers), expected.to(dtype))


class CenterUpdateTests(unittest.TestCase):
    def test_center_update_averages_points_and_reinitializes_empty_clusters(self):
        X = torch.tensor([[0., 2.], [2., 4.], [6., 8.]], dtype=torch.float64)
        labels = torch.tensor([0, 0, 1])
        model = KMeansPlusPlus(3)
        generator = torch.Generator().manual_seed(42)

        centers = model._update_centers(X, labels, generator)

        torch.testing.assert_close(centers[:2], X.new_tensor([[1., 3.], [6., 8.]]))
        self.assertTrue((X == centers[2]).all(dim=1).any().item())


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 5),
    'Euclimix requires an SM75+ GPU',
)
class ClusteringCudaTests(unittest.TestCase):
    def test_fp16_fp64_fit_and_predict_preserve_coordinates_for_fallback(self):
        X = torch.tensor([[128.013], [128.052], [128.040], [128.018]], requires_grad=True)
        initial = torch.tensor([[128.015], [128.048]], requires_grad=True)
        model = KMeansPlusPlus(2, kernel='fp16_fp64', normalize=None, max_iter=1)
        model.fit(X, initial)

        expected = torch.tensor([0, 1, 1, 0])
        torch.testing.assert_close(model.labels_.cpu(), expected)
        torch.testing.assert_close(model.predict(X).cpu(), expected)
        centers = torch.stack((X[[0, 3]].mean(dim=0), X[[1, 2]].mean(dim=0)))
        torch.testing.assert_close(model.cluster_centers_.cpu(), centers)
        self.assertAlmostEqual(model.inertia_, (X - centers[expected]).square().sum().item(), places=8)

    def test_default_kernel_with_all_initialization_methods(self):
        X = torch.tensor([[1., 1.], [1., 2.], [9., 9.], [9., 8.]])
        for method in ('random', 'kmeans++', 'k-means||'):
            with self.subTest(method=method):
                model = KMeansPlusPlus(2, init_method=method, random_state=42)
                labels = model.fit_predict(X)
                torch.testing.assert_close(labels, model.predict(X))
                self.assertNotEqual(labels[0].item(), labels[2].item())

    def test_single_sample_with_empty_cluster_reinitialization(self):
        X = torch.tensor([[2., 3.]])
        model = KMeansPlusPlus(2, random_state=42).fit(X)
        torch.testing.assert_close(model.get_cluster_centers_original_space().cpu(), X.expand(2, -1))
        self.assertEqual(model.inertia_, 0.)
