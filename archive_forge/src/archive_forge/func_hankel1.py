from .functions import defun, defun_wrapped
@defun_wrapped
def hankel1(ctx, n, x, **kwargs):
    return ctx.besselj(n, x, **kwargs) + ctx.j * ctx.bessely(n, x, **kwargs)