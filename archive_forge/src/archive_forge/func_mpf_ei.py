import operator
import math
from .backend import MPZ_ZERO, MPZ_ONE, BACKEND, xrange, exec_
from .libintmath import gcd
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
from .libintmath import ifac
from .gammazeta import mpf_gamma_int, mpf_euler, euler_fixed
def mpf_ei(x, prec, rnd=round_fast, e1=False):
    if e1:
        x = mpf_neg(x)
    sign, man, exp, bc = x
    if e1 and (not sign):
        if x == fzero:
            return finf
        raise ComplexResult('E1(x) for x < 0')
    if man:
        xabs = (0, man, exp, bc)
        xmag = exp + bc
        wp = prec + 20
        can_use_asymp = xmag > wp
        if not can_use_asymp:
            if exp >= 0:
                xabsint = man << exp
            else:
                xabsint = man >> -exp
            can_use_asymp = xabsint > int(wp * 0.693) + 10
        if can_use_asymp:
            if xmag > wp:
                v = fone
            else:
                v = from_man_exp(ei_asymptotic(to_fixed(x, wp), wp), -wp)
            v = mpf_mul(v, mpf_exp(x, wp), wp)
            v = mpf_div(v, x, prec, rnd)
        else:
            wp += 2 * int(to_int(xabs))
            u = to_fixed(x, wp)
            v = ei_taylor(u, wp) + euler_fixed(wp)
            t1 = from_man_exp(v, -wp)
            t2 = mpf_log(xabs, wp)
            v = mpf_add(t1, t2, prec, rnd)
    elif x == fzero:
        v = fninf
    elif x == finf:
        v = finf
    elif x == fninf:
        v = fzero
    else:
        v = fnan
    if e1:
        v = mpf_neg(v)
    return v