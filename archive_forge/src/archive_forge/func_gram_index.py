from .functions import defun, defun_wrapped
def gram_index(ctx, t):
    if t > 10 ** 13:
        wp = 3 * ctx.log(t, 10)
    else:
        wp = 0
    prec = ctx.prec
    try:
        ctx.prec += wp
        h = int(ctx.siegeltheta(t) / ctx.pi)
    finally:
        ctx.prec = prec
    return h