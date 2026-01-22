import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
@constant_memo
def euler_fixed(prec):
    extra = 30
    prec += extra
    p = int(math.log(prec / 4 * math.log(2), 2)) + 1
    n = 2 ** p
    A = U = -p * ln2_fixed(prec)
    B = V = MPZ_ONE << prec
    k = 1
    while 1:
        B = B * n ** 2 // k ** 2
        A = (A * n ** 2 // k + B) // k
        U += A
        V += B
        if max(abs(A), abs(B)) < 100:
            break
        k += 1
    return (U << prec - extra) // V