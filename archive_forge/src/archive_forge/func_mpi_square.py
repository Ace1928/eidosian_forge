from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpi_square(s, prec=0):
    sa, sb = s
    if mpf_ge(sa, fzero):
        a = mpf_mul(sa, sa, prec, round_floor)
        b = mpf_mul(sb, sb, prec, round_ceiling)
    elif mpf_le(sb, fzero):
        a = mpf_mul(sb, sb, prec, round_floor)
        b = mpf_mul(sa, sa, prec, round_ceiling)
    else:
        sa = mpf_neg(sa)
        sa, sb = mpf_min_max([sa, sb])
        a = fzero
        b = mpf_mul(sb, sb, prec, round_ceiling)
    return (a, b)