from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpci_arg(z, prec):
    x, y = z
    return mpi_atan2(y, x, prec)