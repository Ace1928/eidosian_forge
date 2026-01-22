import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_conjugate(z, prec, rnd=round_fast):
    re, im = z
    return (re, mpf_neg(im, prec, rnd))