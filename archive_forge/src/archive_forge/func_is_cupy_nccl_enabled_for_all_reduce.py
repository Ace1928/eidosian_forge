import contextlib
import torch
from vllm.model_executor.parallel_utils import cupy_utils
def is_cupy_nccl_enabled_for_all_reduce():
    """check if CuPy nccl is enabled for all reduce"""
    global _ENABLE_CUPY_FOR_ALL_REDUCE
    return _ENABLE_CUPY_FOR_ALL_REDUCE