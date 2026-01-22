import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_tan(z, prec, rnd=round_fast):
    """Complex tangent. Computed as tan(a+bi) = sin(2a)/M + sinh(2b)/M*i
    where M = cos(2a) + cosh(2b)."""
    a, b = z
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b
    if b == fzero:
        return (mpf_tan(a, prec, rnd), fzero)
    if a == fzero:
        return (fzero, mpf_tanh(b, prec, rnd))
    wp = prec + 15
    a = mpf_shift(a, 1)
    b = mpf_shift(b, 1)
    c, s = mpf_cos_sin(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    mag = mpf_add(c, ch, wp)
    re = mpf_div(s, mag, prec, rnd)
    im = mpf_div(sh, mag, prec, rnd)
    return (re, im)