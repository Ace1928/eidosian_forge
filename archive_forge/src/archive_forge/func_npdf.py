from .functions import defun, defun_wrapped
@defun_wrapped
def npdf(ctx, x, mu=0, sigma=1):
    sigma = ctx.convert(sigma)
    return ctx.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * ctx.sqrt(2 * ctx.pi))