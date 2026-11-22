# mp-kmeans

[![!pypi](https://img.shields.io/pypi/v/mp-kmeans?color=yellowgreen)](https://pypi.org/project/mp-kmeans/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mp-kmeans)](https://pypi.org/project/mp-kmeans/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`mp-kmeans` is a CUDA-accelerated k-means library for PyTorch. Its mixed-precision modes compute squared Euclidean distances with Euclimix and selectively recompute them using the direct distance formula.

## Features

- Squared Euclidean distances for GPU k-means via [Euclimix](https://pypi.org/project/euclimix/).
- Uniform precision modes (`fp16`, `bf16`, `fp32`, `fp64`) and mixed modes (e.g. `fp16_fp32`, `fp16_fp64`).
- GPU center updates with automatic empty-cluster reinitialization.
- Configurable normalization (`standard`, `l2`, `minmax`) for robust behavior on unnormalized datasets.

## Installation

```bash
pip install mp-kmeans
```

Installation resolves `euclimix` and PyTorch through pip. The default kernel requires Python 3.11 or newer, CUDA-enabled PyTorch, and an NVIDIA Turing or newer GPU (SM75+). When pip installs Euclimix from its source distribution, the first GPU call compiles its CUDA extension and requires a CUDA toolkit with `nvcc` and a compatible C++ compiler.

From a source checkout, install the optional dependencies for the precision benchmark with `python -m pip install '.[benchmark]'`.

## Quick Start

```python
from mp_kmeans import KMeansPlusPlus, make_blobs_gpu

X, _ = make_blobs_gpu(
    n_samples=100_000,
    n_features=128,
    n_centers=100,
    cluster_std=1.0,
    random_state=42,
)

model = KMeansPlusPlus(
    n_clusters=100,
    kernel="fp16_fp32",
    kappa=10.0,
    max_iter=300,
    tol=1e-8,
    normalize="standard",
    random_state=42,
)

model.fit(X)
print(model.n_iter_, model.inertia_)
```

## Kernel Modes

- Euclimix modes are `fp16_fp32`, `fp16_fp64`, `bf16_fp32`, `bf16_fp64`, `fp32_fp32` (default), and `fp32_fp64`. The suffix selects recomputation precision, and `kappa` is passed to Euclimix as `rho` (greater than 1 and at most 1,000,000). The Euclimix BF16 modes require Ampere or newer (SM80+).
- PyTorch `cdist` baselines are `fp16`, `bf16`, `fp32`, and `fp64`.
- Expanded-formula baselines are `fp16_uniform`, `bf16_uniform`, `tf32_uniform`, `fp32_uniform`, and `fp64_uniform`. `tf32_uniform` follows PyTorch's CUDA matmul settings.

Version 0.1.1 removed `tf32_fp64`, `fp64_fp16`, `fp64_bf16`, `fp64_tf32`, and `fp64_fp32_gemm`. Use `kernel="fp32"` to select the PyTorch baseline that was the default in version 0.1.0.

## Citation

The paper evaluates synthetic clustering and image segmentation and reports accuracy and convergence comparable to higher-precision baselines on the tested workloads. The reported speedups vary with feature dimension and the number of clusters. [arXiv preprint](https://arxiv.org/abs/2407.12208v4)

```bibtex
@misc{ccl24,
  author = {Erin Carson and Xinye Chen and Xiaobo Liu},
  title = {Mixed-Precision {Lloyd}'s Algorithm for $k$-means Clustering},
  year = {2026},
  month = nov,
  eprint = {2407.12208v4},
  archivePrefix = {arXiv},
  primaryClass = {math.NA},
  url = {https://arxiv.org/abs/2407.12208v4}
}
```
