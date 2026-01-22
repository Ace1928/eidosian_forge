from .functions import defun, defun_wrapped
@defun
def hyperu(ctx, a, b, z, **kwargs):
    a, atype = ctx._convert_param(a)
    b, btype = ctx._convert_param(b)
    z = ctx.convert(z)
    if not z:
        if ctx.re(b) <= 1:
            return ctx.gammaprod([1 - b], [a - b + 1])
        else:
            return ctx.inf + z
    bb = 1 + a - b
    bb, bbtype = ctx._convert_param(bb)
    try:
        orig = ctx.prec
        try:
            ctx.prec += 10
            v = ctx.hypsum(2, 0, (atype, bbtype), [a, bb], -1 / z, maxterms=ctx.prec)
            return v / z ** a
        finally:
            ctx.prec = orig
    except ctx.NoConvergence:
        pass

    def h(a, b):
        w = ctx.sinpi(b)
        T1 = ([ctx.pi, w], [1, -1], [], [a - b + 1, b], [a], [b], z)
        T2 = ([-ctx.pi, w, z], [1, -1, 1 - b], [], [a, 2 - b], [a - b + 1], [2 - b], z)
        return (T1, T2)
    return ctx.hypercomb(h, [a, b], **kwargs)