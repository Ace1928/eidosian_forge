from .functions import defun, defun_wrapped
@defun_wrapped
def ncdf(ctx, x, mu=0, sigma=1):
    a = (x - mu) / (sigma * ctx.sqrt(2))
    if a < 0:
        return ctx.erfc(-a) / 2
    else:
        return (1 + ctx.erf(a)) / 2