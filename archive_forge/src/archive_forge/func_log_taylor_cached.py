import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def log_taylor_cached(x, prec):
    """
    Fixed-point computation of log(x), assuming x in (0.5, 2)
    and prec <= LOG_TAYLOR_PREC.
    """
    n = x >> prec - LOG_TAYLOR_SHIFT
    cached_prec = cache_prec_steps[prec]
    dprec = cached_prec - prec
    if (n, cached_prec) in log_taylor_cache:
        a, log_a = log_taylor_cache[n, cached_prec]
    else:
        a = n << cached_prec - LOG_TAYLOR_SHIFT
        log_a = log_taylor(a, cached_prec, 8)
        log_taylor_cache[n, cached_prec] = (a, log_a)
    a >>= dprec
    log_a >>= dprec
    u = (x - a << prec) // a
    v = (u << prec) // ((MPZ_TWO << prec) + u)
    v2 = v * v >> prec
    v4 = v2 * v2 >> prec
    s0 = v
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
    s = s0 + s1 << 1
    return log_a + s