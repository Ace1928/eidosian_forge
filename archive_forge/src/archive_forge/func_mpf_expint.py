import operator
import math
from .backend import MPZ_ZERO, MPZ_ONE, BACKEND, xrange, exec_
from .libintmath import gcd
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
from .libintmath import ifac
from .gammazeta import mpf_gamma_int, mpf_euler, euler_fixed
def mpf_expint(n, x, prec, rnd=round_fast, gamma=False):
    """
    E_n(x), n an integer, x real

    With gamma=True, computes Gamma(n,x)   (upper incomplete gamma function)

    Returns (real, None) if real, otherwise (real, imag)
    The imaginary part is an optional branch cut term

    """
    sign, man, exp, bc = x
    if not man:
        if gamma:
            if x == fzero:
                if n <= 0:
                    return (finf, None)
                return (mpf_gamma_int(n, prec, rnd), None)
            if x == finf:
                return (fzero, None)
            return (fnan, fnan)
        else:
            if x == fzero:
                if n > 1:
                    return (from_rational(1, n - 1, prec, rnd), None)
                else:
                    return (finf, None)
            if x == finf:
                return (fzero, None)
            return (fnan, fnan)
    n_orig = n
    if gamma:
        n = 1 - n
    wp = prec + 20
    xmag = exp + bc
    if xmag < -10:
        raise NotImplementedError
    nmag = bitcount(abs(n))
    have_imag = n > 0 and sign
    negx = mpf_neg(x)
    if n == 0 or 2 * nmag - xmag < -wp:
        if gamma:
            v = mpf_exp(negx, wp)
            re = mpf_mul(v, mpf_pow_int(x, n_orig - 1, wp), prec, rnd)
        else:
            v = mpf_exp(negx, wp)
            re = mpf_div(v, x, prec, rnd)
    else:
        can_use_asymptotic_series = -3 * wp < n <= 0
        if not can_use_asymptotic_series:
            xi = abs(to_int(x))
            m = min(max(1, xi - n), 2 * wp)
            siz = -n * nmag + (m + n) * bitcount(abs(m + n)) - m * xmag - 144 * m // 100
            tol = -wp - 10
            can_use_asymptotic_series = siz < tol
        if can_use_asymptotic_series:
            r = (-MPZ_ONE << wp + wp) // to_fixed(x, wp)
            m = n
            t = r * m
            s = MPZ_ONE << wp
            while m and t:
                s += t
                m += 1
                t = m * r * t >> wp
            v = mpf_exp(negx, wp)
            if gamma:
                v = mpf_mul(v, mpf_pow_int(x, n_orig - 1, wp), wp)
            else:
                v = mpf_div(v, x, wp)
            re = mpf_mul(v, from_man_exp(s, -wp), prec, rnd)
        elif n == 1:
            re = mpf_neg(mpf_ei(negx, prec, rnd))
        elif n > 0 and n < 3 * wp:
            T1 = mpf_neg(mpf_ei(negx, wp))
            if gamma:
                if n_orig & 1:
                    T1 = mpf_neg(T1)
            else:
                T1 = mpf_mul(T1, mpf_pow_int(negx, n - 1, wp), wp)
            r = t = to_fixed(x, wp)
            facs = [1] * (n - 1)
            for k in range(1, n - 1):
                facs[k] = facs[k - 1] * k
            facs = facs[::-1]
            s = facs[0] << wp
            for k in range(1, n - 1):
                if k & 1:
                    s -= facs[k] * t
                else:
                    s += facs[k] * t
                t = t * r >> wp
            T2 = from_man_exp(s, -wp, wp)
            T2 = mpf_mul(T2, mpf_exp(negx, wp))
            if gamma:
                T2 = mpf_mul(T2, mpf_pow_int(x, n_orig, wp), wp)
            R = mpf_add(T1, T2)
            re = mpf_div(R, from_int(ifac(n - 1)), prec, rnd)
        else:
            raise NotImplementedError
    if have_imag:
        M = from_int(-ifac(n - 1))
        if gamma:
            im = mpf_div(mpf_pi(wp), M, prec, rnd)
            if n_orig & 1:
                im = mpf_neg(im)
        else:
            im = mpf_div(mpf_mul(mpf_pi(wp), mpf_pow_int(negx, n_orig - 1, wp), wp), M, prec, rnd)
        return (re, im)
    else:
        return (re, None)