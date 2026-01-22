import math
from numba import (config, cuda, float32, float64, uint32, int64, uint64,
import numpy as np
@jit(forceobj=_forceobj, looplift=_looplift, nopython=_nopython)
def uint64_to_unit_float32(x):
    """Convert uint64 to float32 value in the range [0.0, 1.0)"""
    x = uint64(x)
    return float32(uint64_to_unit_float64(x))