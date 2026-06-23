"""Unified Python wrapper for CUDA distance and center-update backends."""

from importlib import import_module

_MODULE_NAMES = [
    "mp_kmeans.euclidean_cuda_uniform",
    "mp_kmeans.euclidean_cuda_mix",
    "mp_kmeans.euclidean_cuda_advanced",
    "mp_kmeans.euclidean_cuda_backend",
    "mp_kmeans.center_cuda",
]

_REQUIRED_APIS = [
    "pairwise_euclidean_uniform_fp16",
    "pairwise_euclidean_uniform_bf16",
    "pairwise_euclidean_uniform_tf32",
    "pairwise_euclidean_single",
    "pairwise_euclidean_double",
    "pairwise_euclidean_fp16_fp64",
    "pairwise_euclidean_bf16_fp64",
    "pairwise_euclidean_tf32_fp64",
    "pairwise_euclidean_fp32_fp64",
    "pairwise_euclidean_fp16_fp32",
    "pairwise_euclidean_bf16_fp32",
    "pairwise_euclidean_fp32_fp32",
    "pairwise_euclidean_fp64_fp16",
    "pairwise_euclidean_fp64_bf16",
    "pairwise_euclidean_fp64_tf32",
    "pairwise_euclidean_fp64_fp32_gemm",
    "update_centers_fp64_with_reinit",
    "update_centers_fp32_with_reinit",
]

_loaded_modules = []
_missing_modules = []
for _name in _MODULE_NAMES:
    try:
        _loaded_modules.append(import_module(_name))
    except ModuleNotFoundError:
        _missing_modules.append(_name)


for _api in _REQUIRED_APIS:
    for _module in _loaded_modules:
        if hasattr(_module, _api):
            globals()[_api] = getattr(_module, _api)
            break
    else:
        _loaded_names = [m.__name__ for m in _loaded_modules]
        raise ImportError(
            f"Missing CUDA API '{_api}'. Loaded modules: {_loaded_names}. Missing modules: {_missing_modules}."
        )


__all__ = list(_REQUIRED_APIS)
