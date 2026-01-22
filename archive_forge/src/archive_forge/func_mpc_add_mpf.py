import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_add_mpf(z, x, prec, rnd=round_fast):
    a, b = z
    return (mpf_add(a, x, prec, rnd), b)