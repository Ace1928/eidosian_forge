from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpi_cosh_sinh(x, prec):
    wp = prec + 20
    e1 = mpi_exp(x, wp)
    e2 = mpi_div(mpi_one, e1, wp)
    c = mpi_add(e1, e2, prec)
    s = mpi_sub(e1, e2, prec)
    c = mpi_shift(c, -1)
    s = mpi_shift(s, -1)
    return (c, s)