import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def check_mm_compatible_shapes(f_name, lhs, rhs):
    check(lhs.dim() >= 2 and rhs.dim() >= 2, f'{f_name}(): all inputs involved in the matrix product are expected to be at least 2D, but got lhs.dim() == {lhs.dim()} and rhs.dim() == {rhs.dim()}.')
    m, kl = lhs.shape[-2:]
    kr, n = rhs.shape[-2:]
    check(kl == kr, f"{f_name}(): arguments' sizes involved in the matrix product are not compatible for matrix multiplication, got lhs.shape[-1] == {kl} which is not equal to rhs.shape[-2] == {kr}.")