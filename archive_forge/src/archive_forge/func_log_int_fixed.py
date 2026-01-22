import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def log_int_fixed(n, prec, ln2=None):
    """
    Fast computation of log(n), caching the value for small n,
    intended for zeta sums.
    """
    if n in log_int_cache:
        value, vprec = log_int_cache[n]
        if vprec >= prec:
            return value >> vprec - prec
    wp = prec + 10
    if wp <= LOG_TAYLOR_SHIFT:
        if ln2 is None:
            ln2 = ln2_fixed(wp)
        r = bitcount(n)
        x = n << wp - r
        v = log_taylor_cached(x, wp) + r * ln2
    else:
        v = to_fixed(mpf_log(from_int(n), wp + 5), wp)
    if n < MAX_LOG_INT_CACHE:
        log_int_cache[n] = (v, wp)
    return v >> wp - prec