from __future__ import annotations
from math import ceil, isnan
from packaging.version import Version
import numba
import numpy as np
from numba import cuda
@cuda.jit(device=True)
def cuda_atomic_nanmin(ary, idx, val):
    return cuda.atomic.min(ary, idx, val)