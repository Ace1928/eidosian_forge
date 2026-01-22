from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpci_mul(x, y, prec):
    a, b = x
    c, d = y
    r1 = mpi_mul(a, c)
    r2 = mpi_mul(b, d)
    re = mpi_sub(r1, r2, prec)
    i1 = mpi_mul(a, d)
    i2 = mpi_mul(b, c)
    im = mpi_add(i1, i2, prec)
    return (re, im)