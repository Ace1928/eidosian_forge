import operator
import math
from .backend import MPZ_ZERO, MPZ_ONE, BACKEND, xrange, exec_
from .libintmath import gcd
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
from .libintmath import ifac
from .gammazeta import mpf_gamma_int, mpf_euler, euler_fixed
def mpc_ei(z, prec, rnd=round_fast, e1=False):
    if e1:
        z = mpc_neg(z)
    a, b = z
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b
    if b == fzero:
        if e1:
            x = mpf_neg(mpf_ei(a, prec, rnd))
            if not asign:
                y = mpf_neg(mpf_pi(prec, rnd))
            else:
                y = fzero
            return (x, y)
        else:
            return (mpf_ei(a, prec, rnd), fzero)
    if a != fzero:
        if not aman or not bman:
            return (fnan, fnan)
    wp = prec + 40
    amag = aexp + abc
    bmag = bexp + bbc
    zmag = max(amag, bmag)
    can_use_asymp = zmag > wp
    if not can_use_asymp:
        zabsint = abs(to_int(a)) + abs(to_int(b))
        can_use_asymp = zabsint > int(wp * 0.693) + 20
    try:
        if can_use_asymp:
            if zmag > wp:
                v = (fone, fzero)
            else:
                zre = to_fixed(a, wp)
                zim = to_fixed(b, wp)
                vre, vim = complex_ei_asymptotic(zre, zim, wp)
                v = (from_man_exp(vre, -wp), from_man_exp(vim, -wp))
            v = mpc_mul(v, mpc_exp(z, wp), wp)
            v = mpc_div(v, z, wp)
            if e1:
                v = mpc_neg(v, prec, rnd)
            else:
                x, y = v
                if bsign:
                    v = (mpf_pos(x, prec, rnd), mpf_sub(y, mpf_pi(wp), prec, rnd))
                else:
                    v = (mpf_pos(x, prec, rnd), mpf_add(y, mpf_pi(wp), prec, rnd))
            return v
    except NoConvergence:
        pass
    wp += 2 * int(to_int(mpc_abs(z, 5)))
    zre = to_fixed(a, wp)
    zim = to_fixed(b, wp)
    vre, vim = complex_ei_taylor(zre, zim, wp)
    vre += euler_fixed(wp)
    v = (from_man_exp(vre, -wp), from_man_exp(vim, -wp))
    if e1:
        u = mpc_log(mpc_neg(z), wp)
    else:
        u = mpc_log(z, wp)
    v = mpc_add(v, u, prec, rnd)
    if e1:
        v = mpc_neg(v)
    return v