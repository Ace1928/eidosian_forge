import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def mpf_bernoulli_huge(n, prec, rnd=None):
    wp = prec + 10
    piprec = wp + int(math.log(n, 2))
    v = mpf_gamma_int(n + 1, wp)
    v = mpf_mul(v, mpf_zeta_int(n, wp), wp)
    v = mpf_mul(v, mpf_pow_int(mpf_pi(piprec), -n, wp))
    v = mpf_shift(v, 1 - n)
    if not n & 3:
        v = mpf_neg(v)
    return mpf_pos(v, prec, rnd or round_fast)