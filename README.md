# mp-kmeans


[![!pypi](https://img.shields.io/pypi/v/mp-kmeans?color=yellowgreen)](https://pypi.org/project/mp-kmeans/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mp-kmeans)](https://pypi.org/project/mp-kmeans/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  

A mixed-precision algorithm of $k$-means is designed towards understanding of the low precision arithmetic for Euclidean distance computations.  By performing simulations across data with various settings, we showcase that decreased precision for $k$-means computing only results in a minor increase in sum of squared errors while not necessarily leading to degrading performance regarding clustering results.  

`mp-kmeans` is a CUDA-accelerated mixed-precision implementation of k-means designed for large-scale clustering workloads.
It provides multiple precision paths (FP16/BF16/FP32/FP64 and mixed fallback modes) to balance throughput and numerical stability.

## Features

- Mixed-precision Euclidean distance kernels for GPU k-means.
- Uniform precision modes (`fp16`, `bf16`, `fp32`, `fp64`) and mixed modes (e.g. `fp16_fp32`, `fp16_fp64`).
- CUDA center update kernels with automatic empty-cluster reinitialization.
- Configurable normalization (`standard`, `l2`, `minmax`) for robust behavior on unnormalized datasets.

## Installation

```bash
pip install mp-kmeans
```

> This package targets CUDA-enabled environments and depends on PyTorch with CUDA support.

## Quick Start

```python
import torch
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

- Uniform: `fp16_uniform`, `bf16_uniform`, `tf32_uniform`, `fp32_uniform`, `fp64_uniform`
- Mixed: `fp16_fp32`, `bf16_fp32`, `fp32_fp32`, `fp16_fp64`, `bf16_fp64`, `tf32_fp64`, `fp32_fp64`
- Advanced: `fp64_fp16`, `fp64_bf16`, `fp64_tf32`, `fp64_fp32_gemm`

## Citation

```bibtex
@techreport{ccl24,
  author = "Erin Carson and Xinye Chen and Xiaobo Liu",
  title = "Computing $k$-means in Mixed Precision",
  month = jul,
  year = 2026,
  type = "{ArXiv}:2407.12208 [math.{NA}]",
  url = "https://arxiv.org/abs/2407.12208"
}
```
