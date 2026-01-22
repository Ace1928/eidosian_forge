import functools
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, List, Optional
import sympy
import torch
from ...codecache import cache_dir
from ...config import cuda as inductor_cuda_config
from ...ir import Layout
from .cuda_env import get_cuda_arch, get_cuda_version
def get_accumulator_dtype(input_torch_dtypes: List[torch.dtype]) -> Optional[torch.dtype]:
    """
    Given a list of input torch dtypes, returns the inferred accumulator torch dtype.
    """
    if len(input_torch_dtypes) == 0:
        return None
    torch_dtype = input_torch_dtypes[0]
    for dtype in input_torch_dtypes[1:]:
        if torch_dtype != dtype:
            raise RuntimeError(f'Unmatched input dtypes: torch_dtype={torch_dtype!r}, dtype={dtype!r}')
    if torch_dtype == torch.half:
        if torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction:
            return torch_dtype
        else:
            return torch.float
    if torch_dtype in {torch.bfloat16, torch.float}:
        return torch.float
    raise NotImplementedError(f'Unsupported data type: input_torch_dtypes={input_torch_dtypes!r}')