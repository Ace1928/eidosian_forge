import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_shift(z, n):
    a, b = z
    return (mpf_shift(a, n), mpf_shift(b, n))