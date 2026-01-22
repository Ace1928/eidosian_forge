import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def zeta_array(N, prec):
    """
    zeta(n) = A * pi**n / n! + B

    where A is a rational number (A = Bernoulli number
    for n even) and B is an infinite sum over powers of exp(2*pi).
    (B = 0 for n even).

    TODO: this is currently only used for gamma, but could
    be very useful elsewhere.
    """
    extra = 30
    wp = prec + extra
    zeta_values = [MPZ_ZERO] * (N + 2)
    pi = pi_fixed(wp)
    one = MPZ_ONE << wp
    zeta_values[0] = -one // 2
    f_2pi = mpf_shift(mpf_pi(wp), 1)
    exp_2pi_k = exp_2pi = mpf_exp(f_2pi, wp)
    exps3 = []
    k = 1
    while 1:
        tp = wp - 9 * k
        if tp < 1:
            break
        q1 = mpf_div(fone, mpf_sub(exp_2pi_k, fone, tp), tp)
        q2 = mpf_mul(exp_2pi_k, mpf_mul(q1, q1, tp), tp)
        q1 = to_fixed(q1, wp)
        q2 = to_fixed(q2, wp)
        q2 = k * q2 * pi >> wp
        exps3.append((q1, q2))
        exp_2pi_k = mpf_mul(exp_2pi_k, exp_2pi, wp)
        k += 1
    for n in xrange(3, N + 1, 2):
        s = MPZ_ZERO
        k = 1
        for e1, e2 in exps3:
            if n % 4 == 3:
                t = e1 // k ** n
            else:
                U = (n - 1) // 4
                t = (e1 + e2 // U) // k ** n
            if not t:
                break
            s += t
            k += 1
        zeta_values[n] = -2 * s
    B = [mpf_abs(mpf_bernoulli(k, wp)) for k in xrange(N + 2)]
    pi_pow = fpi = mpf_pow_int(mpf_shift(mpf_pi(wp), 1), 2, wp)
    pi_pow = mpf_div(pi_pow, from_int(4), wp)
    for n in xrange(2, N + 2, 2):
        z = mpf_mul(B[n], pi_pow, wp)
        zeta_values[n] = to_fixed(z, wp)
        pi_pow = mpf_mul(pi_pow, fpi, wp)
        pi_pow = mpf_div(pi_pow, from_int((n + 1) * (n + 2)), wp)
    reciprocal_pi = (one << wp) // pi
    for n in xrange(3, N + 1, 4):
        U = (n - 3) // 4
        s = zeta_values[4 * U + 4] * (4 * U + 7) // 4
        for k in xrange(1, U + 1):
            s -= zeta_values[4 * k] * zeta_values[4 * U + 4 - 4 * k] >> wp
        zeta_values[n] += 2 * s * reciprocal_pi >> wp
    for n in xrange(5, N + 1, 4):
        U = (n - 1) // 4
        s = zeta_values[4 * U + 2] * (2 * U + 1)
        for k in xrange(1, 2 * U + 1):
            s += (-1) ** k * 2 * k * zeta_values[2 * k] * zeta_values[4 * U + 2 - 2 * k] >> wp
        zeta_values[n] += (s * reciprocal_pi >> wp) // (2 * U)
    return [x >> extra for x in zeta_values]