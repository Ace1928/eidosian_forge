from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpci_square(x, prec):
    a, b = x
    re = mpi_sub(mpi_square(a), mpi_square(b), prec)
    im = mpi_mul(a, b, prec)
    im = mpi_shift(im, 1)
    return (re, im)