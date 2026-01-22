import numpy as np
from numba import uint64, uint32, uint16, uint8
from numba.core.extending import register_jitable
from numba.np.random._constants import (UINT32_MAX, UINT64_MAX,
from numba.np.random.generator_core import next_uint32, next_uint64
@register_jitable
def random_bounded_bool_fill(bitgen, low, rng, size, dtype):
    """
    Returns a new array of given size with boolean values.
    """
    buf = 0
    bcnt = 0
    out = np.empty(size, dtype=dtype)
    for i in np.ndindex(size):
        val, bcnt, buf = buffered_bounded_bool(bitgen, low, rng, bcnt, buf)
        out[i] = low + val
    return out