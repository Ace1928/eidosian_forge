import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def atan_newton(x, prec):
    if prec >= 100:
        r = math.atan(int(x >> prec - 53) / 2.0 ** 53)
    else:
        r = math.atan(int(x) / 2.0 ** prec)
    prevp = 50
    r = MPZ(int(r * 2.0 ** 53) >> 53 - prevp)
    extra_p = 50
    for wp in giant_steps(prevp, prec):
        wp += extra_p
        r = r << wp - prevp
        cos, sin = cos_sin_fixed(r, wp)
        tan = (sin << wp) // cos
        a = (tan - rshift(x, prec - wp) << wp) // ((MPZ_ONE << wp) + (tan ** 2 >> wp))
        r = r - a
        prevp = wp
    return rshift(r, prevp - prec)