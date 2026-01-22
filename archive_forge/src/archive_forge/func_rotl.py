import math
from numba import (config, cuda, float32, float64, uint32, int64, uint64,
import numpy as np
@jit(forceobj=_forceobj, looplift=_looplift, nopython=_nopython)
def rotl(x, k):
    """Left rotate x by k bits."""
    x = uint64(x)
    k = uint32(k)
    return x << k | x >> uint32(64 - k)