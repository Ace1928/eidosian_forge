from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpi_str(s, prec):
    sa, sb = s
    dps = prec_to_dps(prec) + 5
    return '[%s, %s]' % (to_str(sa, dps), to_str(sb, dps))