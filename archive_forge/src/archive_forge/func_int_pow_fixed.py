import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def int_pow_fixed(y, n, prec):
    """n-th power of a fixed point number with precision prec

       Returns the power in the form man, exp,
       man * 2**exp ~= y**n
    """
    if n == 2:
        return (y * y, 0)
    bc = bitcount(y)
    exp = 0
    workprec = 2 * (prec + 4 * bitcount(n) + 4)
    _, pm, pe, pbc = fone
    while 1:
        if n & 1:
            pm = pm * y
            pe = pe + exp
            pbc += bc - 2
            pbc = pbc + bctable[int(pm >> pbc)]
            if pbc > workprec:
                pm = pm >> pbc - workprec
                pe += pbc - workprec
                pbc = workprec
            n -= 1
            if not n:
                break
        y = y * y
        exp = exp + exp
        bc = bc + bc - 2
        bc = bc + bctable[int(y >> bc)]
        if bc > workprec:
            y = y >> bc - workprec
            exp += bc - workprec
            bc = workprec
        n = n // 2
    return (pm, pe)