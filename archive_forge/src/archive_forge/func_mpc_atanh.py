import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_atanh(z, prec, rnd=round_fast):
    wp = prec + 15
    a = mpc_add(z, mpc_one, wp)
    b = mpc_sub(mpc_one, z, wp)
    a = mpc_log(a, wp)
    b = mpc_log(b, wp)
    v = mpc_shift(mpc_sub(a, b, wp), -1)
    if v[0] == fnan and mpc_is_inf(z):
        v = (fzero, v[1])
    return v