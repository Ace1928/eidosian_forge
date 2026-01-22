import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_square(z, prec, rnd=round_fast):
    a, b = z
    p = mpf_mul(a, a)
    q = mpf_mul(b, b)
    r = mpf_mul(a, b, prec, rnd)
    re = mpf_sub(p, q, prec, rnd)
    im = mpf_shift(r, 1)
    return (re, im)