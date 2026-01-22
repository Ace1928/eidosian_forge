import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def mpc_psi0(z, prec, rnd=round_fast):
    """
    Computation of the digamma function (psi function of order 0)
    of a complex argument.
    """
    re, im = z
    if im == fzero:
        return (mpf_psi0(re, prec, rnd), fzero)
    wp = prec + 20
    sign, man, exp, bc = re
    if sign and exp + bc > 3:
        c = mpc_cos_pi(z, wp)
        s = mpc_sin_pi(z, wp)
        q = mpc_mul_mpf(mpc_div(c, s, wp), mpf_pi(wp), wp)
        p = mpc_psi0(mpc_sub(mpc_one, z, wp), wp)
        return mpc_sub(p, q, prec, rnd)
    if not sign and bc + exp > wp:
        return mpc_log(mpc_sub(z, mpc_one, wp), prec, rnd)
    w = to_int(re)
    n = int(0.11 * wp) + 2
    s = mpc_zero
    if w < n:
        for k in xrange(w, n):
            s = mpc_sub(s, mpc_reciprocal(z, wp), wp)
            z = mpc_add_mpf(z, fone, wp)
    z = mpc_sub(z, mpc_one, wp)
    s = mpc_add(s, mpc_log(z, wp), wp)
    s = mpc_add(s, mpc_div(mpc_half, z, wp), wp)
    z2 = mpc_square(z, wp)
    t = mpc_one
    prev = mpc_zero
    szprev = fzero
    k = 1
    eps = mpf_shift(fone, -wp + 2)
    while 1:
        t = mpc_mul(t, z2, wp)
        bern = mpf_bernoulli(2 * k, wp)
        term = mpc_mpf_div(bern, mpc_mul_int(t, 2 * k, wp), wp)
        s = mpc_sub(s, term, wp)
        szterm = mpc_abs(term, 10)
        if k > 2 and (mpf_le(szterm, eps) or mpf_le(szprev, szterm)):
            break
        prev = term
        szprev = szterm
        k += 1
    return s