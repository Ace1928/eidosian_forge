from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def hyp1f2(ctx, a1, b1, b2, z, **kwargs):
    return ctx.hyper([a1], [b1, b2], z, **kwargs)