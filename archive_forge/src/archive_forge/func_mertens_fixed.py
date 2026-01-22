import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
@constant_memo
def mertens_fixed(prec):
    wp = prec + 20
    m = 2
    s = mpf_euler(wp)
    while 1:
        t = mpf_zeta_int(m, wp)
        if t == fone:
            break
        t = mpf_log(t, wp)
        t = mpf_mul_int(t, moebius(m), wp)
        t = mpf_div(t, from_int(m), wp)
        s = mpf_add(s, t)
        m += 1
    return to_fixed(s, prec)