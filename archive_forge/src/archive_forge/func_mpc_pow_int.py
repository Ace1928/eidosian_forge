import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_pow_int(z, n, prec, rnd=round_fast):
    a, b = z
    if b == fzero:
        return (mpf_pow_int(a, n, prec, rnd), fzero)
    if a == fzero:
        v = mpf_pow_int(b, n, prec, rnd)
        n %= 4
        if n == 0:
            return (v, fzero)
        elif n == 1:
            return (fzero, v)
        elif n == 2:
            return (mpf_neg(v), fzero)
        elif n == 3:
            return (fzero, mpf_neg(v))
    if n == 0:
        return mpc_one
    if n == 1:
        return mpc_pos(z, prec, rnd)
    if n == 2:
        return mpc_square(z, prec, rnd)
    if n == -1:
        return mpc_reciprocal(z, prec, rnd)
    if n < 0:
        return mpc_reciprocal(mpc_pow_int(z, -n, prec + 4), prec, rnd)
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b
    if asign:
        aman = -aman
    if bsign:
        bman = -bman
    de = aexp - bexp
    abs_de = abs(de)
    exact_size = n * (abs_de + max(abc, bbc))
    if exact_size < 10000:
        if de > 0:
            aman <<= de
            aexp = bexp
        else:
            bman <<= -de
            bexp = aexp
        re, im = complex_int_pow(aman, bman, n)
        re = from_man_exp(re, int(n * aexp), prec, rnd)
        im = from_man_exp(im, int(n * bexp), prec, rnd)
        return (re, im)
    return mpc_exp(mpc_mul_int(mpc_log(z, prec + 10), n, prec + 10), prec, rnd)