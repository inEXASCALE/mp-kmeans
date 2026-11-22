# mp-kmeans

[![CI](https://github.com/inEXASCALE/mp-kmeans/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/inEXASCALE/mp-kmeans/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/mp-kmeans?label=PyPI&logo=pypi&logoColor=white&color=3775A9)](https://pypi.org/project/mp-kmeans/) [![Python 3.11–3.14](https://img.shields.io/badge/Python-3.11%20%E2%80%93%203.14-3776AB?logo=python&logoColor=white)](https://github.com/inEXASCALE/mp-kmeans#installation) [![CUDA SM75+](https://img.shields.io/badge/CUDA-SM75%2B-517600?logo=nvidia&logoColor=white)](https://github.com/inEXASCALE/mp-kmeans#installation) [![arXiv:2407.12208](https://img.shields.io/badge/arXiv-2407.12208-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2407.12208v4) [![License: MIT](https://img.shields.io/badge/License-MIT-64748B)](https://github.com/inEXASCALE/mp-kmeans/blob/main/LICENSE)

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

## Testing

GitHub Actions builds the wheel and source distribution, checks the package metadata, and runs CPU tests against the installed wheel on Python 3.11–3.14.

The CUDA tests are available as a manual workflow on `main`. The workflow runs on a self-hosted Linux x86-64 runner labeled `gpu` with Python 3.11 or later, an NVIDIA SM75+ GPU, and a CUDA toolkit with `nvcc` on `PATH`. [CUDA workflow](https://github.com/inEXASCALE/mp-kmeans/actions/workflows/cuda.yml)

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
