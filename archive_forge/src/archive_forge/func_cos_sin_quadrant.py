from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def cos_sin_quadrant(x, wp):
    sign, man, exp, bc = x
    if x == fzero:
        return (fone, fzero, 0)
    c, s = mpf_cos_sin(x, wp)
    t, n, wp_ = mod_pi2(man, exp, exp + bc, 15)
    if sign:
        n = -1 - n
    return (c, s, n)