from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpci_div(x, y, prec):
    a, b = x
    c, d = y
    wp = prec + 20
    m1 = mpi_square(c)
    m2 = mpi_square(d)
    m = mpi_add(m1, m2, wp)
    re = mpi_add(mpi_mul(a, c), mpi_mul(b, d), wp)
    im = mpi_sub(mpi_mul(b, c), mpi_mul(a, d), wp)
    re = mpi_div(re, m, prec)
    im = mpi_div(im, m, prec)
    return (re, im)