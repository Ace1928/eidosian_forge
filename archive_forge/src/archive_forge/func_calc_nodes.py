import math
from ..libmp.backend import xrange
def calc_nodes(self, degree, prec, verbose=False):
    """
        Calculates the abscissas and weights for Gauss-Legendre
        quadrature of degree of given degree (actually `3 \\cdot 2^m`).
        """
    ctx = self.ctx
    epsilon = ctx.ldexp(1, -prec - 8)
    orig = ctx.prec
    ctx.prec = int(prec * 1.5)
    if degree == 1:
        x = ctx.sqrt(ctx.mpf(3) / 5)
        w = ctx.mpf(5) / 9
        nodes = [(-x, w), (ctx.zero, ctx.mpf(8) / 9), (x, w)]
        ctx.prec = orig
        return nodes
    nodes = []
    n = 3 * 2 ** (degree - 1)
    upto = n // 2 + 1
    for j in xrange(1, upto):
        r = ctx.mpf(math.cos(math.pi * (j - 0.25) / (n + 0.5)))
        while 1:
            t1, t2 = (1, 0)
            for j1 in xrange(1, n + 1):
                t3, t2, t1 = (t2, t1, ((2 * j1 - 1) * r * t1 - (j1 - 1) * t2) / j1)
            t4 = n * (r * t1 - t2) / (r ** 2 - 1)
            a = t1 / t4
            r = r - a
            if abs(a) < epsilon:
                break
        x = r
        w = 2 / ((1 - r ** 2) * t4 ** 2)
        if verbose and j % 30 == 15:
            print('Computing nodes (%i of %i)' % (j, upto))
        nodes.append((x, w))
        nodes.append((-x, w))
    ctx.prec = orig
    return nodes