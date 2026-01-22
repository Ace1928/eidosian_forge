import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_mul_imag_mpf(z, x, prec, rnd=round_fast):
    """
    Multiply the mpc value z by I*x where x is an mpf value.
    """
    a, b = z
    re = mpf_neg(mpf_mul(b, x, prec, rnd))
    im = mpf_mul(a, x, prec, rnd)
    return (re, im)