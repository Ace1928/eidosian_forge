import importlib
import importlib.metadata
import os
import warnings
from functools import lru_cache
import torch
from packaging import version
from packaging.version import parse
from .environment import parse_flag_from_env, str_to_bool
from .versions import compare_versions, is_torch_version
def is_bf16_available(ignore_tpu=False):
    """Checks if bf16 is supported, optionally ignoring the TPU"""
    if is_torch_xla_available(check_is_tpu=True):
        return not ignore_tpu
    if is_cuda_available():
        return torch.cuda.is_bf16_supported()
    return True