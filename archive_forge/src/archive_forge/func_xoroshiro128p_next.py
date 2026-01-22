import math
from numba import (config, cuda, float32, float64, uint32, int64, uint64,
import numpy as np
@jit(forceobj=_forceobj, looplift=_looplift, nopython=_nopython)
def xoroshiro128p_next(states, index):
    """Return the next random uint64 and advance the RNG in states[index].

    :type states: 1D array, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type index: int64
    :param index: offset in states to update
    :rtype: uint64
    """
    index = int64(index)
    s0 = states[index]['s0']
    s1 = states[index]['s1']
    result = s0 + s1
    s1 ^= s0
    states[index]['s0'] = uint64(rotl(s0, uint32(55))) ^ s1 ^ s1 << uint32(14)
    states[index]['s1'] = uint64(rotl(s1, uint32(36)))
    return result