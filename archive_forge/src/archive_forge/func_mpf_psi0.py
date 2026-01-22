import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def mpf_psi0(x, prec, rnd=round_fast):
    """
    Computation of the digamma function (psi function of order 0)
    of a real argument.
    """
    sign, man, exp, bc = x
    wp = prec + 10
    if not man:
        if x == finf:
            return x
        if x == fninf or x == fnan:
            return fnan
    if x == fzero or (exp >= 0 and sign):
        raise ValueError('polygamma pole')
    if exp + bc < -5:
        v = mpf_psi0(mpf_add(x, fone, prec, rnd), prec, rnd)
        return mpf_sub(v, mpf_div(fone, x, wp, rnd), prec, rnd)
    if sign and exp + bc > 3:
        c, s = mpf_cos_sin_pi(x, wp)
        q = mpf_mul(mpf_div(c, s, wp), mpf_pi(wp), wp)
        p = mpf_psi0(mpf_sub(fone, x, wp), wp)
        return mpf_sub(p, q, prec, rnd)
    if not sign and bc + exp > wp:
        return mpf_log(mpf_sub(x, fone, wp), prec, rnd)
    m = to_int(x)
    n = int(0.11 * wp) + 2
    s = MPZ_ZERO
    x = to_fixed(x, wp)
    one = MPZ_ONE << wp
    if m < n:
        for k in xrange(m, n):
            s -= (one << wp) // x
            x += one
    x -= one
    s += to_fixed(mpf_log(from_man_exp(x, -wp, wp), wp), wp)
    s += (one << wp) // (2 * x)
    x2 = x * x >> wp
    t = one
    prev = 0
    k = 1
    while 1:
        t = t * x2 >> wp
        bsign, bman, bexp, bbc = mpf_bernoulli(2 * k, wp)
        offset = bexp + 2 * wp
        if offset >= 0:
            term = (bman << offset) // (t * (2 * k))
        else:
            term = (bman >> -offset) // (t * (2 * k))
        if k & 1:
            s -= term
        else:
            s += term
        if k > 2 and term >= prev:
            break
        prev = term
        k += 1
    return from_man_exp(s, -wp, wp, rnd)