from __future__ import annotations
from math import ceil, isnan
from packaging.version import Version
import numba
import numpy as np
from numba import cuda
@cuda.jit
def cuda_row_min_in_place(ret, other):
    """CUDA equivalent of row_min_in_place.
    """
    ny, nx, ncat = ret.shape
    x, y, cat = cuda.grid(3)
    if x < nx and y < ny and (cat < ncat):
        if other[y, x, cat] > -1 and (ret[y, x, cat] == -1 or other[y, x, cat] < ret[y, x, cat]):
            ret[y, x, cat] = other[y, x, cat]