import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def mpf_exp(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if man:
        mag = bc + exp
        wp = prec + 14
        if sign:
            man = -man
        if prec > 600 and exp >= 0:
            e = mpf_e(wp + int(1.45 * mag))
            return mpf_pow_int(e, man << exp, prec, rnd)
        if mag < -wp:
            return mpf_perturb(fone, sign, prec, rnd)
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
        man = exp_basecase(t, wp)
        return from_man_exp(man, n - wp, prec, rnd)
    if not exp:
        return fone
    if x == fninf:
        return fzero
    return x