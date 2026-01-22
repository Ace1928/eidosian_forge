from __future__ import annotations
from math import ceil, isnan
from packaging.version import Version
import numba
import numpy as np
from numba import cuda
@cuda.jit
def masked_clip_2d_kernel(data, mask, lower, upper):
    i, j = cuda.grid(2)
    maxi, maxj = data.shape
    if i >= 0 and i < maxi and (j >= 0) and (j < maxj) and (not mask[i, j]):
        cuda_atomic_nanmax(data, (i, j), lower)
        cuda_atomic_nanmin(data, (i, j), upper)