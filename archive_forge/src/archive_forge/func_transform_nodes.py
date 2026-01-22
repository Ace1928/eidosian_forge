import math
from ..libmp.backend import xrange
def transform_nodes(self, nodes, a, b, verbose=False):
    """
        Rescale standardized nodes (for `[-1, 1]`) to a general
        interval `[a, b]`. For a finite interval, a simple linear
        change of variables is used. Otherwise, the following
        transformations are used:

        .. math ::

            \\lbrack a, \\infty \\rbrack : t = \\frac{1}{x} + (a-1)

            \\lbrack -\\infty, b \\rbrack : t = (b+1) - \\frac{1}{x}

            \\lbrack -\\infty, \\infty \\rbrack : t = \\frac{x}{\\sqrt{1-x^2}}

        """
    ctx = self.ctx
    a = ctx.convert(a)
    b = ctx.convert(b)
    one = ctx.one
    if (a, b) == (-one, one):
        return nodes
    half = ctx.mpf(0.5)
    new_nodes = []
    if ctx.isinf(a) or ctx.isinf(b):
        if (a, b) == (ctx.ninf, ctx.inf):
            p05 = -half
            for x, w in nodes:
                x2 = x * x
                px1 = one - x2
                spx1 = px1 ** p05
                x = x * spx1
                w *= spx1 / px1
                new_nodes.append((x, w))
        elif a == ctx.ninf:
            b1 = b + 1
            for x, w in nodes:
                u = 2 / (x + one)
                x = b1 - u
                w *= half * u ** 2
                new_nodes.append((x, w))
        elif b == ctx.inf:
            a1 = a - 1
            for x, w in nodes:
                u = 2 / (x + one)
                x = a1 + u
                w *= half * u ** 2
                new_nodes.append((x, w))
        elif a == ctx.inf or b == ctx.ninf:
            return [(x, -w) for x, w in self.transform_nodes(nodes, b, a, verbose)]
        else:
            raise NotImplementedError
    else:
        C = (b - a) / 2
        D = (b + a) / 2
        for x, w in nodes:
            new_nodes.append((D + C * x, C * w))
    return new_nodes