import operator
import math
from .backend import MPZ_ZERO, MPZ_ONE, BACKEND, xrange, exec_
from .libintmath import gcd
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
from .libintmath import ifac
from .gammazeta import mpf_gamma_int, mpf_euler, euler_fixed
def ei_asymptotic(x, prec):
    one = MPZ_ONE << prec
    x = t = (one << prec) // x
    s = one + x
    k = 2
    while t:
        t = k * t * x >> prec
        s += t
        k += 1
    return s