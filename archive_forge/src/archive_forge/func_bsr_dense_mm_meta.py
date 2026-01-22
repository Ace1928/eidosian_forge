import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def bsr_dense_mm_meta(M, K, N, Ms, Ks, GROUP_SIZE_ROW=None, num_warps=None, num_stages=None, **extra):
    if {num_warps, num_stages, GROUP_SIZE_ROW} == {None}:
        device_name = torch.cuda.get_device_name()
        meta = get_meta('bsr_dense_mm', (M, K, N, Ms, Ks), device_name, version=(0, torch.float16, 0.5))
        if meta is not None:
            meta.update(**extra)
            return meta
    GROUP_SIZE_ROW = GROUP_SIZE_ROW or 4
    num_stages = num_stages or 1
    num_warps = num_warps or 4
    return dict(GROUP_SIZE_ROW=GROUP_SIZE_ROW, num_stages=num_stages, num_warps=num_warps, **extra)