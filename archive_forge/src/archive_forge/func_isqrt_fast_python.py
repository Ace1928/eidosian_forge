import math
from bisect import bisect
from .backend import xrange
from .backend import BACKEND, gmpy, sage, sage_utils, MPZ, MPZ_ONE, MPZ_ZERO
def isqrt_fast_python(x):
    """
    Fast approximate integer square root, computed using division-free
    Newton iteration for large x. For random integers the result is almost
    always correct (floor(sqrt(x))), but is 1 ulp too small with a roughly
    0.1% probability. If x is very close to an exact square, the answer is
    1 ulp wrong with high probability.

    With 0 guard bits, the largest error over a set of 10^5 random
    inputs of size 1-10^5 bits was 3 ulp. The use of 10 guard bits
    almost certainly guarantees a max 1 ulp error.
    """
    if x < _1_800:
        y = int(x ** 0.5)
        if x >= _1_100:
            y = y + x // y >> 1
            if x >= _1_200:
                y = y + x // y >> 1
                if x >= _1_400:
                    y = y + x // y >> 1
        return y
    bc = bitcount(x)
    guard_bits = 10
    x <<= 2 * guard_bits
    bc += 2 * guard_bits
    bc += bc & 1
    hbc = bc // 2
    startprec = min(50, hbc)
    r = int(2.0 ** (2 * startprec) * (x >> bc - 2 * startprec) ** (-0.5))
    pp = startprec
    for p in giant_steps(startprec, hbc):
        r2 = r * r >> 2 * pp - p
        xr2 = (x >> bc - p) * r2 >> p
        r = r * ((3 << p) - xr2) >> pp + 1
        pp = p
    return r * (x >> hbc) >> p + guard_bits