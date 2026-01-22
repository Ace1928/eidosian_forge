import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def mpf_gamma_int(n, prec, rnd=round_fast):
    if n < SMALL_FACTORIAL_CACHE_SIZE:
        return mpf_pos(small_factorial_cache[n - 1], prec, rnd)
    return mpf_gamma(from_int(n), prec, rnd)