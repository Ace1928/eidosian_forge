import numpy as np
from numba import uint64, uint32, uint16, uint8
from numba.core.extending import register_jitable
from numba.np.random._constants import (UINT32_MAX, UINT64_MAX,
from numba.np.random.generator_core import next_uint32, next_uint64
@register_jitable
def bounded_lemire_uint64(bitgen, rng):
    """
    Generates a random unsigned 64 bit integer bounded
    within a given interval using Lemire's rejection.
    """
    rng_excl = uint64(rng) + uint64(1)
    assert rng != 18446744073709551615
    x = next_uint64(bitgen)
    leftover = uint64(x) * uint64(rng_excl)
    if leftover < rng_excl:
        threshold = (UINT64_MAX - rng) % rng_excl
        while leftover < threshold:
            x = next_uint64(bitgen)
            leftover = uint64(x) * uint64(rng_excl)
    x0 = x & uint64(4294967295)
    x1 = x >> 32
    rng_excl0 = rng_excl & uint64(4294967295)
    rng_excl1 = rng_excl >> 32
    w0 = x0 * rng_excl0
    t = x1 * rng_excl0 + (w0 >> 32)
    w1 = t & uint64(4294967295)
    w2 = t >> 32
    w1 += x0 * rng_excl1
    m1 = x1 * rng_excl1 + w2 + (w1 >> 32)
    return m1