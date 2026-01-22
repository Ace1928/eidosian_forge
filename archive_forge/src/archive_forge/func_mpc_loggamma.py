import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def mpc_loggamma(z, prec, rnd='d'):
    a, b = z
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b
    if b == fzero and asign:
        re = mpf_gamma(a, prec, rnd, 3)
        n = -aman >> -aexp
        im = mpf_mul_int(mpf_pi(prec + 10), n, prec, rnd)
        return (re, im)
    return mpc_gamma(z, prec, rnd, 3)