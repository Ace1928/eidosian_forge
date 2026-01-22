import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_acosh(z, prec, rnd=round_fast):
    a, b = mpc_acos(z, prec, rnd)
    if b[0] or b == fzero:
        return (mpf_neg(b), a)
    else:
        return (b, mpf_neg(a))