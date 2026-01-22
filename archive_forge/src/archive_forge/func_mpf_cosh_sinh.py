import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def mpf_cosh_sinh(x, prec, rnd=round_fast, tanh=0):
    """Simultaneously compute (cosh(x), sinh(x)) for real x"""
    sign, man, exp, bc = x
    if not man and exp:
        if tanh:
            if x == finf:
                return fone
            if x == fninf:
                return fnone
            return fnan
        if x == finf:
            return (finf, finf)
        if x == fninf:
            return (finf, fninf)
        return (fnan, fnan)
    mag = exp + bc
    wp = prec + 14
    if mag < -4:
        if mag < -wp:
            if tanh:
                return mpf_perturb(x, 1 - sign, prec, rnd)
            cosh = mpf_perturb(fone, 0, prec, rnd)
            sinh = mpf_perturb(x, sign, prec, rnd)
            return (cosh, sinh)
        wp += -mag
    if mag > 10:
        if 3 * (1 << mag - 1) > wp:
            if tanh:
                return mpf_perturb([fone, fnone][sign], 1 - sign, prec, rnd)
            c = s = mpf_shift(mpf_exp(mpf_abs(x), prec, rnd), -1)
            if sign:
                s = mpf_neg(s)
            return (c, s)
    if mag > 1:
        wpmod = wp + mag
        offset = exp + wpmod
        if offset >= 0:
            t = man << offset
        else:
            t = man >> -offset
        lg2 = ln2_fixed(wpmod)
        n, t = divmod(t, lg2)
        n = int(n)
        t >>= mag
    else:
        offset = exp + wp
        if offset >= 0:
            t = man << offset
        else:
            t = man >> -offset
        n = 0
    a, b = exp_expneg_basecase(t, wp)
    cosh = a + (b >> 2 * n)
    sinh = a - (b >> 2 * n)
    if sign:
        sinh = -sinh
    if tanh:
        man = (sinh << wp) // cosh
        return from_man_exp(man, -wp, prec, rnd)
    else:
        cosh = from_man_exp(cosh, n - wp - 1, prec, rnd)
        sinh = from_man_exp(sinh, n - wp - 1, prec, rnd)
        return (cosh, sinh)