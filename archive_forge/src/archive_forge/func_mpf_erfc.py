import operator
import math
from .backend import MPZ_ZERO, MPZ_ONE, BACKEND, xrange, exec_
from .libintmath import gcd
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
from .libintmath import ifac
from .gammazeta import mpf_gamma_int, mpf_euler, euler_fixed
def mpf_erfc(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if not man:
        if x == fzero:
            return fone
        if x == finf:
            return fzero
        if x == fninf:
            return ftwo
        return fnan
    wp = prec + 20
    mag = bc + exp
    wp += max(0, 2 * mag)
    regular_erf = sign or mag < 2
    if regular_erf or not erfc_check_series(x, wp):
        if regular_erf:
            return mpf_sub(fone, mpf_erf(x, prec + 10, negative_rnd[rnd]), prec, rnd)
        n = to_int(x) + 1
        return mpf_sub(fone, mpf_erf(x, prec + int(n ** 2 * 1.44) + 10), prec, rnd)
    s = term = MPZ_ONE << wp
    term_prev = 0
    t = 2 * to_fixed(x, wp) ** 2 >> wp
    k = 1
    while 1:
        term = (term * (2 * k - 1) << wp) // t
        if k > 4 and term > term_prev or not term:
            break
        if k & 1:
            s -= term
        else:
            s += term
        term_prev = term
        k += 1
    s = (s << wp) // sqrt_fixed(pi_fixed(wp), wp)
    s = from_man_exp(s, -wp, wp)
    z = mpf_exp(mpf_neg(mpf_mul(x, x, wp), wp), wp)
    y = mpf_div(mpf_mul(z, s, wp), x, prec, rnd)
    return y