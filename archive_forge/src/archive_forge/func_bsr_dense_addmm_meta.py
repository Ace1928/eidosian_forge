import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def bsr_dense_addmm_meta(M, K, N, Ms, Ks, beta, alpha, SPLIT_N=None, GROUP_SIZE_ROW=None, num_warps=None, num_stages=None, dtype=None, **extra):
    if dtype is None:
        dtype = torch.float16
    if {SPLIT_N, num_warps, num_stages, GROUP_SIZE_ROW} == {None}:
        device_name = torch.cuda.get_device_name()
        meta = get_meta('bsr_dense_addmm', (M, K, N, Ms, Ks, beta == 0, beta == 1, alpha == 1), device_name, version=(0, dtype, 0.5))
        if meta is not None:
            meta.update(**extra)
            return meta
    SPLIT_N = SPLIT_N or 1
    GROUP_SIZE_ROW = GROUP_SIZE_ROW or 4
    num_stages = num_stages or 1
    num_warps = num_warps or 4
    return dict(SPLIT_N=SPLIT_N, GROUP_SIZE_ROW=GROUP_SIZE_ROW, num_stages=num_stages, num_warps=num_warps, **extra)