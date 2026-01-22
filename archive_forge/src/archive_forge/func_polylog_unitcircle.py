from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
def polylog_unitcircle(ctx, n, z):
    tol = +ctx.eps
    if n > 1:
        l = ctx.zero
        logz = ctx.ln(z)
        logmz = ctx.one
        m = 0
        while 1:
            if n - m != 1:
                term = ctx.zeta(n - m) * logmz / ctx.fac(m)
                if term and abs(term) < tol:
                    break
                l += term
            logmz *= logz
            m += 1
        l += ctx.ln(z) ** (n - 1) / ctx.fac(n - 1) * (ctx.harmonic(n - 1) - ctx.ln(-ctx.ln(z)))
    elif n < 1:
        l = ctx.fac(-n) * (-ctx.ln(z)) ** (n - 1)
        logz = ctx.ln(z)
        logkz = ctx.one
        k = 0
        while 1:
            b = ctx.bernoulli(k - n + 1)
            if b:
                term = b * logkz / (ctx.fac(k) * (k - n + 1))
                if abs(term) < tol:
                    break
                l -= term
            logkz *= logz
            k += 1
    else:
        raise ValueError
    if ctx._is_real_type(z) and z < 0:
        l = ctx._re(l)
    return l