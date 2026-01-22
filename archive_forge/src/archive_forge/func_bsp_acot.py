import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def bsp_acot(q, a, b, hyperbolic):
    if b - a == 1:
        a1 = MPZ(2 * a + 3)
        if hyperbolic or a & 1:
            return (MPZ_ONE, a1 * q ** 2, a1)
        else:
            return (-MPZ_ONE, a1 * q ** 2, a1)
    m = (a + b) // 2
    p1, q1, r1 = bsp_acot(q, a, m, hyperbolic)
    p2, q2, r2 = bsp_acot(q, m, b, hyperbolic)
    return (q2 * p1 + r1 * p2, q1 * q2, r1 * r2)