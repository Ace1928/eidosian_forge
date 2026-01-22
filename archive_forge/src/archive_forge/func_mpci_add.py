from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpci_add(x, y, prec):
    a, b = x
    c, d = y
    return (mpi_add(a, c, prec), mpi_add(b, d, prec))