import operator
import math
from .backend import MPZ_ZERO, MPZ_ONE, BACKEND, xrange, exec_
from .libintmath import gcd
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
from .libintmath import ifac
from .gammazeta import mpf_gamma_int, mpf_euler, euler_fixed
def mpf_besseljn(n, x, prec, rounding=round_fast):
    prec += 50
    negate = n < 0 and n & 1
    mag = x[2] + x[3]
    n = abs(n)
    wp = prec + 20 + n * bitcount(n)
    if mag < 0:
        wp -= n * mag
    x = to_fixed(x, wp)
    x2 = x ** 2 >> wp
    if not n:
        s = t = MPZ_ONE << wp
    else:
        s = t = x ** n // ifac(n) >> (n - 1) * wp + n
    k = 1
    while t:
        t = t * x2 // (-4 * k * (k + n)) >> wp
        s += t
        k += 1
    if negate:
        s = -s
    return from_man_exp(s, -wp, prec, rounding)