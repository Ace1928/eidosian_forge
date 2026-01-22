from ..libmp.backend import xrange
import math
import cmath
@defun_wrapped
def cot(ctx, z):
    return ctx.one / ctx.tan(z)