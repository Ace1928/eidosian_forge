from .functions import defun, defun_wrapped
def separate_my_zero(ctx, my_zero_number, zero_number_block, T, V, prec):
    """If we know which zero of this block is mine,
    the function separates the zero"""
    variations = 0
    v0 = V[0]
    for k in range(1, len(V)):
        v1 = V[k]
        if v0 * v1 < 0:
            variations += 1
            if variations == my_zero_number:
                k0 = k
                leftv = v0
                rightv = v1
        v0 = v1
    t1 = T[k0]
    t0 = T[k0 - 1]
    ctx.prec = prec
    wpz = wpzeros(my_zero_number * ctx.log(my_zero_number))
    guard = 4 * ctx.mag(my_zero_number)
    precs = [ctx.prec + 4]
    index = 0
    while precs[0] > 2 * wpz:
        index += 1
        precs = [precs[0] // 2 + 3 + 2 * index] + precs
    ctx.prec = precs[0] + guard
    r = ctx.findroot(lambda x: ctx.siegelz(x), (t0, t1), solver='illinois', verbose=False)
    z = ctx.mpc(0.5, r)
    for prec in precs[1:]:
        ctx.prec = prec + guard
        znew = z - ctx.zeta(z) / ctx.zeta(z, derivative=1)
        z = ctx.mpc(0.5, ctx.im(znew))
    return ctx.im(z)