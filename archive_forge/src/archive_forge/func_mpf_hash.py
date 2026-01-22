import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def mpf_hash(s):
    if sys.version_info >= (3, 2):
        ssign, sman, sexp, sbc = s
        if not sman:
            if s == fnan:
                return sys.hash_info.nan
            if s == finf:
                return sys.hash_info.inf
            if s == fninf:
                return -sys.hash_info.inf
        h = sman % HASH_MODULUS
        if sexp >= 0:
            sexp = sexp % HASH_BITS
        else:
            sexp = HASH_BITS - 1 - (-1 - sexp) % HASH_BITS
        h = (h << sexp) % HASH_MODULUS
        if ssign:
            h = -h
        if h == -1:
            h = -2
        return int(h)
    else:
        try:
            return hash(to_float(s, strict=1))
        except OverflowError:
            return hash(s)