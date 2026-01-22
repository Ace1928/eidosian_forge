from ..libmp.backend import xrange
import math
import cmath
@defun
def im(ctx, x):
    x = ctx.convert(x)
    if hasattr(x, 'imag'):
        return x.imag
    return ctx.zero