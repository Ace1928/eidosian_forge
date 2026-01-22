import math
from ..libmp.backend import xrange
def sum_next(self, f, nodes, degree, prec, previous, verbose=False):
    """
        Step sum for tanh-sinh quadrature of degree `m`. We exploit the
        fact that half of the abscissas at degree `m` are precisely the
        abscissas from degree `m-1`. Thus reusing the result from
        the previous level allows a 2x speedup.
        """
    h = self.ctx.mpf(2) ** (-degree)
    if previous:
        S = previous[-1] / (h * 2)
    else:
        S = self.ctx.zero
    S += self.ctx.fdot(((w, f(x)) for x, w in nodes))
    return h * S