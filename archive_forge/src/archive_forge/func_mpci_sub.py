from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpci_sub(x, y, prec):
    a, b = x
    c, d = y
    return (mpi_sub(a, c, prec), mpi_sub(b, d, prec))