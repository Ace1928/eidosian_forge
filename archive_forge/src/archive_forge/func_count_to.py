from .functions import defun, defun_wrapped
def count_to(ctx, t, T, V):
    count = 0
    vold = V[0]
    told = T[0]
    tnew = T[1]
    k = 1
    while tnew < t:
        vnew = V[k]
        if vold * vnew < 0:
            count += 1
        vold = vnew
        k += 1
        tnew = T[k]
    a = ctx.siegelz(t)
    if a * vold < 0:
        count += 1
    return count