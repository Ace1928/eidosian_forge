import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def borwein_coefficients(n):
    if n in borwein_cache:
        return borwein_cache[n]
    ds = [MPZ_ZERO] * (n + 1)
    d = MPZ_ONE
    s = ds[0] = MPZ_ONE
    for i in range(1, n + 1):
        d = d * 4 * (n + i - 1) * (n - i + 1)
        d //= 2 * i * (2 * i - 1)
        s += d
        ds[i] = s
    borwein_cache[n] = ds
    return ds