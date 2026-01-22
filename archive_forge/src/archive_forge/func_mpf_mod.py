import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def mpf_mod(s, t, prec, rnd=round_fast):
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    if not sman and sexp or (not tman and texp):
        return fnan
    if ssign == tsign and texp > sexp + sbc:
        return s
    if tman == 1 and sexp > texp + tbc:
        return fzero
    base = min(sexp, texp)
    sman = (-1) ** ssign * sman
    tman = (-1) ** tsign * tman
    man = (sman << sexp - base) % (tman << texp - base)
    if man >= 0:
        sign = 0
    else:
        man = -man
        sign = 1
    return normalize(sign, man, base, bitcount(man), prec, rnd)