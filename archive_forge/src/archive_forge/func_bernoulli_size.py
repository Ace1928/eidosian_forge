import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def bernoulli_size(n):
    """Accurately estimate the size of B_n (even n > 2 only)"""
    lgn = math.log(n, 2)
    return int(2.326 + 0.5 * lgn + n * (lgn - 4.094))