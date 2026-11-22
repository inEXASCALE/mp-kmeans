"""
KMeans++ with Multi-Precision Support and Normalization
"""
import math
import time
from typing import Optional, Tuple, Literal

import torch
from euclimix import direct_squared_distances, pairwise_squared_distances


_MIXED_KERNELS = {
    'fp16_fp32', 'bf16_fp32', 'fp32_fp32',
    'fp16_fp64', 'bf16_fp64', 'fp32_fp64',
}
_BASELINE_DTYPES = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'fp32': torch.float32,
    'fp64': torch.float64,
    'fp16_uniform': torch.float16,
    'bf16_uniform': torch.bfloat16,
    'tf32_uniform': torch.float32,
    'fp32_uniform': torch.float32,
    'fp64_uniform': torch.float64,
}


class KMeansPlusPlus:
    """
    KMeans++ clustering with multiple precision modes and normalization
    
    Normalization options:
    - None: No normalization
    - 'l2': L2 normalization (unit vectors)
    - 'standard': Standardization (zero mean, unit variance)
    - 'minmax': Min-max scaling to [0, 1]

    K-means|| controls:
    - km_parallel_oversampling: oversampling factor l (None -> 2*k)
    - km_parallel_rounds: number of sampling rounds (None -> ceil(log2(N+1)))
    """
    
    def __init__(
        self,
        n_clusters: int,
        kernel: Literal[
            'fp16', 'bf16', 'fp32', 'fp64',
            'fp16_uniform', 'bf16_uniform', 'tf32_uniform', 'fp32_uniform', 'fp64_uniform',
            'fp16_fp32', 'bf16_fp32', 'fp32_fp32',
            'fp16_fp64', 'bf16_fp64', 'fp32_fp64',
        ] = 'fp32_fp32',
        kappa: float = 5.0,
        max_iter: int = 300,
        tol: float = 1e-4,
        normalize: Optional[Literal['l2', 'standard', 'minmax']] = 'standard',
        random_state: Optional[int] = None,
        init_method: Literal['kmeans++', 'random', 'k-means||'] = 'random',
        verbose: bool = False,
        km_parallel_oversampling: Optional[int] = None,
        km_parallel_rounds: Optional[int] = None,
    ):
        if kernel not in _MIXED_KERNELS and kernel not in _BASELINE_DTYPES:
            supported = sorted(_MIXED_KERNELS | _BASELINE_DTYPES.keys())
            raise ValueError(f"Unknown kernel {kernel!r}. Supported kernels: {', '.join(supported)}")
        self.init_method = init_method
        self.n_clusters = n_clusters
        self.kernel = kernel
        self.kappa = kappa
        self.max_iter = max_iter
        self.tol = tol
        self.normalize = normalize
        self.random_state = random_state
        self.verbose = verbose
        self.km_parallel_oversampling = km_parallel_oversampling
        self.km_parallel_rounds = km_parallel_rounds
        
        # Normalization parameters (learned from training data)
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None
        
        # Results
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.timing_ = {}
        
    def _get_dtype(self):
        return _BASELINE_DTYPES.get(self.kernel, torch.float32)
    
    def _normalize_data(self, X: torch.Tensor, fit: bool = True) -> torch.Tensor:
        """
        Normalize data according to self.normalize
        
        Args:
            X: Input data (N, D)
            fit: If True, learn normalization parameters from X
        
        Returns:
            Normalized data
        """
        if self.normalize is None:
            return X

        if self.kernel in _MIXED_KERNELS and X.dtype not in (torch.float32, torch.float64):
            X = X.float()

        if self.normalize == 'l2':
            # L2 normalization (unit vectors)
            norms = torch.norm(X, p=2, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-12)  # Avoid division by zero
            X_norm = X / norms
            
        elif self.normalize == 'standard':
            # Standardization (zero mean, unit variance)
            if fit:
                self.mean_ = X.mean(dim=0, keepdim=True)
                self.std_ = X.std(dim=0, keepdim=True, correction=int(X.shape[0] > 1))
                self.std_ = torch.clamp(self.std_, min=1e-8)  # Avoid division by zero
            
            X_norm = (X - self.mean_) / self.std_
            
        elif self.normalize == 'minmax':
            # Min-max scaling to [0, 1]
            if fit:
                self.min_ = X.min(dim=0, keepdim=True).values
                self.max_ = X.max(dim=0, keepdim=True).values
                range_ = self.max_ - self.min_
                range_ = torch.clamp(range_, min=1e-8)
                self.range_ = range_
            
            X_norm = (X - self.min_) / self.range_
            
        else:
            raise ValueError(f"Unknown normalization: {self.normalize}")
        
        return X_norm
    
    def _compute_distances(self, X: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        if self.kernel in _MIXED_KERNELS:
            precision, correction = self.kernel.split('_')
            return pairwise_squared_distances(
                X.float(), centers.float(), precision=precision, correction=correction, rho=self.kappa,
            )

        dtype = _BASELINE_DTYPES[self.kernel]
        X, centers = X.to(dtype), centers.to(dtype)
        low_precision = dtype in (torch.float16, torch.bfloat16)
        if self.kernel.endswith('_uniform'):
            # The uniform baselines store the norms and cross products in the
            # selected dtype before forming the expanded squared distance.
            points = X.float() if low_precision else X
            refs = centers.float() if low_precision else centers
            x_norm = points.square().sum(dim=1, keepdim=True).to(dtype).to(points.dtype)
            c_norm = refs.square().sum(dim=1).to(dtype).to(points.dtype)
            distances = (X @ centers.T).to(points.dtype)
            distances.mul_(-2).add_(x_norm).add_(c_norm)
            distances = torch.fmax(distances, distances.new_zeros(())).to(dtype)
        elif low_precision:
            distances = torch.cdist(X, centers, compute_mode='use_mm_for_euclid_dist').square()
        else:
            distances = torch.cdist(X, centers).square()
        return distances.float() if low_precision else distances

    def _update_centers(
        self, X: torch.Tensor, labels: torch.Tensor, generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        counts = torch.bincount(labels, minlength=self.n_clusters)
        sums = X.new_zeros((self.n_clusters, X.shape[1]))
        sums.index_add_(0, labels, X)
        centers = sums / counts.clamp_min(1).unsqueeze(1)

        empty = (counts == 0).nonzero(as_tuple=True)[0]
        if empty.numel():
            rows = torch.randint(X.shape[0], (empty.numel(),), device=X.device, generator=generator)
            centers[empty] = X[rows]
        return centers

    @torch.no_grad()
    def fit(self, X: torch.Tensor, C: Optional[torch.Tensor] = None) -> 'KMeansPlusPlus':
        """Fit the model on X. Optional initial centers C are in normalized space."""
        X = X.detach()
        if X.ndim != 2 or min(X.shape) == 0:
            raise ValueError("X must be a nonempty matrix of shape (n_samples, n_features)")
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be positive")
        if not X.is_cuda:
            X = X.cuda()

        t_norm_start = time.time()
        X_normalized = self._normalize_data(X, fit=True)
        t_norm = time.time() - t_norm_start
        if self.verbose and self.normalize:
            print(f"Normalization ({self.normalize}): {t_norm:.4f}s")

        target_dtype = self._get_dtype()
        points_for_distances = X_normalized.to(target_dtype)
        update_dtype = torch.float64 if target_dtype == torch.float64 else torch.float32
        points_for_update = X_normalized.to(update_dtype)
        generator = self._make_generator(X.device) if self.random_state is not None else None

        if C is None:
            t0 = time.time()
            centers = self._init_centers(X_normalized).to(target_dtype)
            self.timing_['init'] = time.time() - t0
        else:
            shape = (self.n_clusters, X.shape[1])
            if C.shape != shape:
                raise ValueError(f"Provided centers C must have shape {shape}")
            centers = C.detach().to(device=X.device, dtype=target_dtype)
            self.timing_['init'] = 0.0

        self.timing_['normalization'] = t_norm
        self.timing_['iterations'] = []
        for iteration in range(self.max_iter):
            iter_start = time.time()
            labels = self._compute_distances(points_for_distances, centers).argmin(dim=1)
            new_centers = self._update_centers(points_for_update, labels, generator).to(target_dtype)
            center_shift = torch.norm(new_centers.float() - centers.float()).item()
            centers = new_centers

            iter_time = time.time() - iter_start
            self.timing_['iterations'].append(iter_time)
            if self.verbose:
                print(f"Iteration {iteration + 1}: shift={center_shift:.6f}, time={iter_time:.4f}s")
            if center_shift < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break

        self.n_iter_ = len(self.timing_['iterations'])
        min_dists, self.labels_ = self._compute_distances(points_for_distances, centers).min(dim=1)
        self.cluster_centers_ = centers
        self.inertia_ = min_dists.sum().item()
        self.timing_['total'] = sum(self.timing_['iterations']) + self.timing_['init'] + t_norm
        return self

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict cluster labels for X."""
        if self.cluster_centers_ is None:
            raise RuntimeError("Model not fitted yet")

        X = X.detach().to(self.cluster_centers_.device)
        X_normalized = self._normalize_data(X, fit=False)
        return self._compute_distances(X_normalized, self.cluster_centers_).argmin(dim=1)

    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        """Fit and return cluster labels"""
        self.fit(X)
        return self.labels_
    
    def get_cluster_centers_original_space(self) -> torch.Tensor:
        """
        Get cluster centers in original (un-normalized) space
        
        Returns:
            Centers in original space
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Model not fitted yet")
        
        centers = self.cluster_centers_.float()
        
        if self.normalize == 'standard':
            # Inverse standardization
            centers = centers * self.std_ + self.mean_
        elif self.normalize == 'minmax':
            # Inverse min-max scaling
            centers = centers * self.range_ + self.min_
        elif self.normalize == 'l2':
            # Cannot invert L2 normalization uniquely
            import warnings
            warnings.warn("Cannot invert L2 normalization. Returning normalized centers.")
        
        return centers


    def _init_centers(self, X: torch.Tensor) -> torch.Tensor:
        method = self.init_method.lower().replace(" ", "")

        if method == "random":
            return self._init_centers_random(X)
        if method in ("kmeans++", "k-means++"):
            return self._init_centers_kmeanspp(X)
        if method in ("kmeans||", "k-means||", "kmeansparallel"):
            return self._init_centers_kmeans_parallel(X)

        raise ValueError(
            f"Unknown init_method={self.init_method!r}. "
            f"Expected one of: 'random', 'kmeans++', 'kmeans||'."
        )
        
    def _make_generator(self, device: torch.device) -> torch.Generator:
        g = torch.Generator(device=device)
        if self.random_state is not None:
            g.manual_seed(int(self.random_state))
        return g

    def _sq_euclidean_distances_to_point(self, X: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.kernel in _MIXED_KERNELS:
            return direct_squared_distances(X, x.unsqueeze(0), precision='fp32').squeeze(1)
        diff = X - x.unsqueeze(0)
        return diff.square().sum(dim=1)

    def _assign_to_centers(self, X: torch.Tensor, centers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dists = self._sq_euclidean_distances_to_centers(X, centers)
        min_sq_dists, labels = dists.min(dim=1)
        return labels, min_sq_dists

    def _weighted_kmeanspp_on_candidates(
        self,
        candidates: torch.Tensor,
        weights: torch.Tensor,
        n_select: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        M, D = candidates.shape
        device = candidates.device
        dtype = candidates.dtype

        C32 = candidates if candidates.dtype == torch.float32 else candidates.float()
        weights32 = weights if weights.dtype == torch.float32 else weights.float()

        centers = torch.empty(n_select, D, device=device, dtype=dtype)

        total_w = weights32.sum()
        if total_w <= 0:
            first_idx = torch.randint(0, M, (1,), device=device, generator=generator).item()
        else:
            first_idx = torch.multinomial(weights32 / total_w, 1, generator=generator).item()

        centers[0] = candidates[first_idx]

        min_sq_dists = self._sq_euclidean_distances_to_point(C32, C32[first_idx])

        chosen = torch.zeros(M, dtype=torch.bool, device=device)
        chosen[first_idx] = True
        min_sq_dists[chosen] = 0.0

        for k in range(1, n_select):
            sampling_weights = weights32 * min_sq_dists
            sampling_weights[chosen] = 0.0
            total = sampling_weights.sum()

            if total <= 0:
                remaining = (~chosen).nonzero(as_tuple=False).flatten()
                if remaining.numel() == 0:
                    next_idx = torch.randint(0, M, (1,), device=device, generator=generator).item()
                else:
                    pos = torch.randint(0, remaining.numel(), (1,), device=device, generator=generator).item()
                    next_idx = remaining[pos].item()
            else:
                probs = sampling_weights / total
                next_idx = torch.multinomial(probs, 1, generator=generator).item()

            centers[k] = candidates[next_idx]
            chosen[next_idx] = True

            new_sq_dists = self._sq_euclidean_distances_to_point(C32, C32[next_idx])
            min_sq_dists = torch.minimum(min_sq_dists, new_sq_dists)
            min_sq_dists[chosen] = 0.0

        return centers

    def _init_centers_random(self, X: torch.Tensor) -> torch.Tensor:
        N, D = X.shape
        device = X.device
        dtype = X.dtype
        g = self._make_generator(device)

        centers = torch.empty(self.n_clusters, D, device=device, dtype=dtype)

        if self.n_clusters <= N:
            perm = torch.randperm(N, generator=g, device=device)
            chosen_idx = perm[:self.n_clusters]
        else:
            chosen_idx = torch.randint(0, N, (self.n_clusters,), generator=g, device=device)

        centers.copy_(X[chosen_idx])

        if self.verbose:
            print(f"Random initialization (kernel: {self.kernel}, normalize: {self.normalize})...")

        return centers

    def _init_centers_kmeanspp(self, X: torch.Tensor) -> torch.Tensor:
        N, D = X.shape
        device = X.device
        dtype = X.dtype
        g = self._make_generator(device)

        X32 = X if X.dtype == torch.float32 else X.float()
        centers = torch.empty(self.n_clusters, D, device=device, dtype=dtype)

        if self.verbose:
            print(f"KMeans++ initialization (kernel: {self.kernel}, normalize: {self.normalize})...")

        first_idx = torch.randint(0, N, (1,), generator=g, device=device).item()
        centers[0] = X[first_idx]

        chosen = torch.zeros(N, dtype=torch.bool, device=device)
        chosen[first_idx] = True

        min_sq_dists = self._sq_euclidean_distances_to_point(X32, X32[first_idx])
        min_sq_dists[chosen] = 0.0

        for k in range(1, self.n_clusters):
            weights = min_sq_dists.clone()
            weights[chosen] = 0.0
            total = weights.sum()

            if total <= 0:
                remaining = (~chosen).nonzero(as_tuple=False).flatten()
                if remaining.numel() == 0:
                    next_idx = torch.randint(0, N, (1,), generator=g, device=device).item()
                else:
                    pos = torch.randint(0, remaining.numel(), (1,), generator=g, device=device).item()
                    next_idx = remaining[pos].item()
            else:
                probs = weights / total
                next_idx = torch.multinomial(probs, 1, generator=g).item()

            centers[k] = X[next_idx]
            chosen[next_idx] = True

            new_sq_dists = self._sq_euclidean_distances_to_point(X32, X32[next_idx])
            min_sq_dists = torch.minimum(min_sq_dists, new_sq_dists)
            min_sq_dists[chosen] = 0.0

        return centers
        
    def _sq_euclidean_distances_to_centers(self, X: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Compute squared distances in FP32 for k-means|| seeding."""
        X32, C32 = X.float(), centers.float()
        if self.kernel in _MIXED_KERNELS:
            return pairwise_squared_distances(X32, C32, precision='fp32', correction='fp32')
        return torch.cdist(X32, C32).square()
        
    def _init_centers_kmeans_parallel(self, X: torch.Tensor) -> torch.Tensor:
        """
        Tensorized k-means|| initialization under squared Euclidean distance.
    
        Procedure:
        1. Sample one initial center uniformly.
        2. Run R rounds of parallel D^2-sampling with oversampling factor l.
        3. Assign weights to candidate centers.
        4. Recluster the candidates down to K centers using weighted k-means++ seeding.
        """
        N, D = X.shape
        device = X.device
        dtype = X.dtype
        g = self._make_generator(device)
    
        X32 = X if X.dtype == torch.float32 else X.float()
    
        l = self.km_parallel_oversampling
        if l is None:
            l = max(2 * self.n_clusters, 1)
    
        R = self.km_parallel_rounds
        if R is None:
            R = max(1, int(math.ceil(math.log2(N + 1))))
    
        if self.verbose:
            print(
                f"KMeans|| initialization (kernel: {self.kernel}, normalize: {self.normalize}, "
                f"oversampling={l}, rounds={R})..."
            )
    
        # First center
        first_idx = torch.randint(0, N, (1,), generator=g, device=device)
        candidate_indices = first_idx.clone()   # shape: (1,)
    
        # Initial nearest squared distances to the first center
        min_sq_dists = self._sq_euclidean_distances_to_point(X32, X32[first_idx.item()])
        min_sq_dists[first_idx] = 0.0
    
        # R rounds of parallel D^2-sampling
        for t in range(R):
            phi = min_sq_dists.sum()
    
            if not torch.isfinite(phi):
                raise ValueError("Encountered non-finite potential in k-means|| initialization")
    
            if phi <= 0:
                sampled_indices = torch.empty(0, dtype=torch.long, device=device)
            else:
                probs = (l * min_sq_dists / phi).clamp_(max=1.0)
                sampled_mask = torch.rand(N, device=device, generator=g) < probs
                sampled_indices = torch.nonzero(sampled_mask, as_tuple=False).flatten()
    
            if sampled_indices.numel() > 0:
                # Tensorized candidate accumulation
                candidate_indices = torch.cat([candidate_indices, sampled_indices], dim=0)
    
                # Update min distances using only newly sampled centers
                new_centers = X32[sampled_indices]  # (m, D)
                new_sq_dists = self._sq_euclidean_distances_to_centers(X32, new_centers)  # (N, m)
                new_min_sq_dists = new_sq_dists.min(dim=1).values
                min_sq_dists = torch.minimum(min_sq_dists, new_min_sq_dists)
    
                # already selected candidates have zero distance to candidate set
                min_sq_dists[candidate_indices] = 0.0
    
            if self.verbose:
                print(f"  Round {t + 1}/{R}: sampled {sampled_indices.numel()} candidates")
    
        # Deduplicate candidate indices
        candidate_indices = torch.unique(candidate_indices, sorted=False)
    
        candidates = X[candidate_indices]   # original dtype for returned centers
    
        if candidates.shape[0] == 0:
            # Extremely defensive fallback
            return self._init_centers_random(X)
    
        # Assign weights to candidates
        labels, _ = self._assign_to_centers(X, candidates)
        weights = torch.bincount(labels, minlength=candidates.shape[0]).to(device=device, dtype=torch.float32)
    
        # If candidates are fewer than required centers, pad by random sampling
        if candidates.shape[0] < self.n_clusters:
            if self.verbose:
                print(
                    f"  Only {candidates.shape[0]} unique candidates for K={self.n_clusters}; "
                    f"padding with random points."
                )
    
            extra_needed = self.n_clusters - candidates.shape[0]
            extra_idx = torch.randint(0, N, (extra_needed,), generator=g, device=device)
            extra_centers = X[extra_idx]
            return torch.cat([candidates, extra_centers], dim=0)
    
        # Recluster candidates down to K centers using weighted k-means++ seeding
        final_centers = self._weighted_kmeanspp_on_candidates(
            candidates=candidates,
            weights=weights,
            n_select=self.n_clusters,
            generator=g,
        )
    
        return final_centers

def make_blobs_gpu(
    n_samples: int,
    n_features: int,
    n_centers: int,
    cluster_std: float = 1.0,
    center_box: Tuple[float, float] = (-10.0, 10.0),
    random_state: Optional[int] = None,
    return_centers: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate Gaussian blobs on GPU
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_centers: Number of cluster centers
        cluster_std: Standard deviation of clusters
        center_box: Bounding box for centers
        random_state: Random seed
        return_centers: If True, also return true centers
    
    Returns:
        X: Data (n_samples, n_features)
        y: True labels (n_samples,)
        centers: True centers (n_centers, n_features) [if return_centers=True]
    """
    if random_state is not None:
        torch.manual_seed(random_state)
    
    # Generate centers
    centers = torch.rand(n_centers, n_features, device='cuda') * (center_box[1] - center_box[0]) + center_box[0]
    
    # Generate samples
    samples_per_center = n_samples // n_centers
    remainder = n_samples % n_centers
    
    X_list = []
    y_list = []
    
    for i in range(n_centers):
        n = samples_per_center + (1 if i < remainder else 0)
        X_cluster = centers[i:i+1] + torch.randn(n, n_features, device='cuda') * cluster_std
        y_cluster = torch.full((n,), i, device='cuda', dtype=torch.long)
        X_list.append(X_cluster)
        y_list.append(y_cluster)
    
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    
    # Shuffle
    perm = torch.randperm(n_samples, device='cuda')
    X = X[perm]
    y = y[perm]
    
    if return_centers:
        return X, y, centers
    return X, y
