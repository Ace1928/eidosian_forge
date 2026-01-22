import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def bspe(a, b):
    """
    Sum series for exp(1)-1 between a, b, returning the result
    as an exact fraction (p, q).
    """
    if b - a == 1:
        return (MPZ_ONE, MPZ(b))
    m = (a + b) // 2
    p1, q1 = bspe(a, m)
    p2, q2 = bspe(m, b)
    return (p1 * q2 + p2, q1 * q2)