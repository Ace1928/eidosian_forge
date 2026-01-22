from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpci_pos(x, prec):
    a, b = x
    return (mpi_pos(a, prec), mpi_pos(b, prec))