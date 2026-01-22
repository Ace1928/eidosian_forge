import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def mpf_asin(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if bc + exp > 0 and x not in (fone, fnone):
        raise ComplexResult('asin(x) is real only for -1 <= x <= 1')
    wp = prec + 15
    a = mpf_mul(x, x)
    b = mpf_add(fone, mpf_sqrt(mpf_sub(fone, a, wp), wp), wp)
    c = mpf_div(x, b, wp)
    return mpf_shift(mpf_atan(c, prec, rnd), 1)