import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_acos(z, prec, rnd=round_fast):
    return acos_asin(z, prec, rnd, 0)