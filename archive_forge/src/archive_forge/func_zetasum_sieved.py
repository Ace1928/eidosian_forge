import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def zetasum_sieved(critical_line, sre, sim, a, n, wp):
    if a < 1:
        raise ValueError('a cannot be less than 1')
    sieve, primes, mult = primesieve(a + n)
    basic_powers = {}
    one = MPZ_ONE << wp
    one_2wp = MPZ_ONE << 2 * wp
    wp2 = wp + wp
    ln2 = ln2_fixed(wp)
    pi2 = pi_fixed(wp - 1)
    for p in primes:
        if p * 2 > a + n:
            break
        log = log_int_fixed(p, wp, ln2)
        cos, sin = cos_sin_fixed(-sim * log >> wp, wp, pi2)
        if critical_line:
            u = one_2wp // isqrt_fast(p << wp2)
        else:
            u = exp_fixed(-sre * log >> wp, wp)
        pre = u * cos >> wp
        pim = u * sin >> wp
        basic_powers[p] = [(pre, pim)]
        tre, tim = (pre, pim)
        for m in range(1, int(math.log(a + n, p) + 0.01) + 1):
            tre, tim = (pre * tre - pim * tim >> wp, pim * tre + pre * tim >> wp)
            basic_powers[p].append((tre, tim))
    xre = MPZ_ZERO
    xim = MPZ_ZERO
    if a == 1:
        xre += one
    aa = max(a, 2)
    for k in xrange(aa, a + n + 1):
        p = sieve[k]
        if p in basic_powers:
            m = mult[k]
            tre, tim = basic_powers[p][m - 1]
            while 1:
                k //= p ** m
                if k == 1:
                    break
                p = sieve[k]
                m = mult[k]
                pre, pim = basic_powers[p][m - 1]
                tre, tim = (pre * tre - pim * tim >> wp, pim * tre + pre * tim >> wp)
        else:
            log = log_int_fixed(k, wp, ln2)
            cos, sin = cos_sin_fixed(-sim * log >> wp, wp, pi2)
            if critical_line:
                u = one_2wp // isqrt_fast(k << wp2)
            else:
                u = exp_fixed(-sre * log >> wp, wp)
            tre = u * cos >> wp
            tim = u * sin >> wp
        xre += tre
        xim += tim
    return (xre, xim)