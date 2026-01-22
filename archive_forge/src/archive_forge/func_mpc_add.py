import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_add(z, w, prec, rnd=round_fast):
    a, b = z
    c, d = w
    return (mpf_add(a, c, prec, rnd), mpf_add(b, d, prec, rnd))