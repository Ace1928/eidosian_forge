import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def mpf_zeta(s, prec, rnd=round_fast, alt=0):
    sign, man, exp, bc = s
    if not man:
        if s == fzero:
            if alt:
                return fhalf
            else:
                return mpf_neg(fhalf)
        if s == finf:
            return fone
        return fnan
    wp = prec + 20
    if not sign and exp + bc > math.log(wp, 2) + 2:
        return mpf_perturb(fone, alt, prec, rnd)
    elif exp >= 0:
        if alt:
            if s == fone:
                return mpf_ln2(prec, rnd)
            z = mpf_zeta_int(to_int(s), wp, negative_rnd[rnd])
            q = mpf_sub(fone, mpf_pow(ftwo, mpf_sub(fone, s, wp), wp), wp)
            return mpf_mul(z, q, prec, rnd)
        else:
            return mpf_zeta_int(to_int(s), prec, rnd)
    if sign:
        if alt:
            q = mpf_sub(fone, mpf_pow(ftwo, mpf_sub(fone, s, wp), wp), wp)
            return mpf_mul(mpf_zeta(s, wp), q, prec, rnd)
        y = mpf_sub(fone, s, 10 * wp)
        a = mpf_gamma(y, wp)
        b = mpf_zeta(y, wp)
        c = mpf_sin_pi(mpf_shift(s, -1), wp)
        wp2 = wp + max(0, exp + bc)
        pi = mpf_pi(wp + wp2)
        d = mpf_div(mpf_pow(mpf_shift(pi, 1), s, wp2), pi, wp2)
        return mpf_mul(a, mpf_mul(b, mpf_mul(c, d, wp), wp), prec, rnd)
    r = mpf_sub(fone, s, wp)
    asign, aman, aexp, abc = mpf_abs(r)
    pole_dist = -2 * (aexp + abc)
    if pole_dist > wp:
        if alt:
            return mpf_ln2(prec, rnd)
        else:
            q = mpf_neg(mpf_div(fone, r, wp))
            return mpf_add(q, mpf_euler(wp), prec, rnd)
    else:
        wp += max(0, pole_dist)
    t = MPZ_ZERO
    n = int(wp / 2.54 + 5)
    d = borwein_coefficients(n)
    t = MPZ_ZERO
    sf = to_fixed(s, wp)
    ln2 = ln2_fixed(wp)
    for k in xrange(n):
        u = -sf * log_int_fixed(k + 1, wp, ln2) >> wp
        eman = exp_fixed(u, wp, ln2)
        w = (d[k] - d[n]) * eman
        if k & 1:
            t -= w
        else:
            t += w
    t = t // -d[n]
    t = from_man_exp(t, -wp, wp)
    if alt:
        return mpf_pos(t, prec, rnd)
    else:
        q = mpf_sub(fone, mpf_pow(ftwo, mpf_sub(fone, s, wp), wp), wp)
        return mpf_div(t, q, prec, rnd)