from ..libmp.backend import xrange
import math
import cmath
@defun_wrapped
def polyexp(ctx, s, z):
    if ctx.isinf(z) or ctx.isinf(s) or ctx.isnan(z) or ctx.isnan(s):
        return z ** s
    if z == 0:
        return z * s
    if s == 0:
        return ctx.expm1(z)
    if s == 1:
        return ctx.exp(z) * z
    if s == 2:
        return ctx.exp(z) * z * (z + 1)
    return _polyexp(ctx, s, z)