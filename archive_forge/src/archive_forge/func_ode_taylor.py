from bisect import bisect
from ..libmp.backend import xrange
def ode_taylor(ctx, derivs, x0, y0, tol_prec, n):
    h = tol = ctx.ldexp(1, -tol_prec)
    dim = len(y0)
    xs = [x0]
    ys = [y0]
    x = x0
    y = y0
    orig = ctx.prec
    try:
        ctx.prec = orig * (1 + n)
        for i in range(n):
            fxy = derivs(x, y)
            y = [y[i] + h * fxy[i] for i in xrange(len(y))]
            x += h
            xs.append(x)
            ys.append(y)
        ser = [[] for d in range(dim)]
        for j in range(n + 1):
            s = [0] * dim
            b = (-1) ** (j & 1)
            k = 1
            for i in range(j + 1):
                for d in range(dim):
                    s[d] += b * ys[i][d]
                b = b * (j - k + 1) // -k
                k += 1
            scale = h ** (-j) / ctx.fac(j)
            for d in range(dim):
                s[d] = s[d] * scale
                ser[d].append(s[d])
    finally:
        ctx.prec = orig
    radius = ctx.one
    for ts in ser:
        if ts[-1]:
            radius = min(radius, ctx.nthroot(tol / abs(ts[-1]), n))
    radius /= 2
    return (ser, x0 + radius)