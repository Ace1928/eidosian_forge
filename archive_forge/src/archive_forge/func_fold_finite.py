from ..libmp.backend import xrange
from .calculus import defun
def fold_finite(ctx, f, intervals):
    if not intervals:
        return f
    indices = [v[0] for v in intervals]
    points = [v[1] for v in intervals]
    ranges = [xrange(a, b + 1) for a, b in points]

    def g(*args):
        args = list(args)
        s = ctx.zero
        for xs in cartesian_product(ranges):
            for dim, x in zip(indices, xs):
                args[dim] = ctx.mpf(x)
            s += f(*args)
        return s
    return g