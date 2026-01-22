import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def atan_taylor(x, prec):
    n = x >> prec - ATAN_TAYLOR_SHIFT
    a, atan_a = atan_taylor_get_cached(n, prec)
    d = x - a
    s0 = v = (d << prec) // ((a ** 2 >> prec) + (a * d >> prec) + (MPZ_ONE << prec))
    v2 = v ** 2 >> prec
    v4 = v2 * v2 >> prec
    s1 = v // 3
    v = v * v4 >> prec
    k = 5
    while v:
        s0 += v // k
        k += 2
        s1 += v // k
        v = v * v4 >> prec
        k += 2
    s1 = s1 * v2 >> prec
    s = s0 - s1
    return atan_a + s