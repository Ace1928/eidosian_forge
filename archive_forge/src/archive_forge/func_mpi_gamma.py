from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpi_gamma(z, prec, type=0):
    a, b = z
    wp = prec + 20
    if type == 1:
        return mpi_gamma(mpi_add(z, mpi_one, wp), prec, 0)
    if mpf_gt(a, gamma_min_b):
        if type == 0:
            c = mpf_gamma(a, prec, round_floor)
            d = mpf_gamma(b, prec, round_ceiling)
        elif type == 2:
            c = mpf_rgamma(b, prec, round_floor)
            d = mpf_rgamma(a, prec, round_ceiling)
        elif type == 3:
            c = mpf_loggamma(a, prec, round_floor)
            d = mpf_loggamma(b, prec, round_ceiling)
    elif mpf_gt(a, fzero) and mpf_lt(b, gamma_min_a):
        if type == 0:
            c = mpf_gamma(b, prec, round_floor)
            d = mpf_gamma(a, prec, round_ceiling)
        elif type == 2:
            c = mpf_rgamma(a, prec, round_floor)
            d = mpf_rgamma(b, prec, round_ceiling)
        elif type == 3:
            c = mpf_loggamma(b, prec, round_floor)
            d = mpf_loggamma(a, prec, round_ceiling)
    else:
        znew = mpi_add(z, mpi_one, wp)
        if type == 0:
            return mpi_div(mpi_gamma(znew, prec + 2, 0), z, prec)
        if type == 2:
            return mpi_mul(mpi_gamma(znew, prec + 2, 2), z, prec)
        if type == 3:
            return mpi_sub(mpi_gamma(znew, prec + 2, 3), mpi_log(z, prec + 2), prec)
    return (c, d)