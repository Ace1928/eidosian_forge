import operator
import math
from .backend import MPZ_ZERO, MPZ_ONE, BACKEND, xrange, exec_
from .libintmath import gcd
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
from .libintmath import ifac
from .gammazeta import mpf_gamma_int, mpf_euler, euler_fixed
def mpc_besseljn(n, z, prec, rounding=round_fast):
    negate = n < 0 and n & 1
    n = abs(n)
    origprec = prec
    zre, zim = z
    mag = max(zre[2] + zre[3], zim[2] + zim[3])
    prec += 20 + n * bitcount(n) + abs(mag)
    if mag < 0:
        prec -= n * mag
    zre = to_fixed(zre, prec)
    zim = to_fixed(zim, prec)
    z2re = zre ** 2 - zim ** 2 >> prec
    z2im = zre * zim >> prec - 1
    if not n:
        sre = tre = MPZ_ONE << prec
        sim = tim = MPZ_ZERO
    else:
        re, im = complex_int_pow(zre, zim, n)
        sre = tre = re // ifac(n) >> (n - 1) * prec + n
        sim = tim = im // ifac(n) >> (n - 1) * prec + n
    k = 1
    while abs(tre) + abs(tim) > 3:
        p = -4 * k * (k + n)
        tre, tim = (tre * z2re - tim * z2im, tim * z2re + tre * z2im)
        tre = tre // p >> prec
        tim = tim // p >> prec
        sre += tre
        sim += tim
        k += 1
    if negate:
        sre = -sre
        sim = -sim
    re = from_man_exp(sre, -prec, origprec, rounding)
    im = from_man_exp(sim, -prec, origprec, rounding)
    return (re, im)