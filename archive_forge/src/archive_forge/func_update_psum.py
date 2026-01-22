from ..libmp.backend import xrange
from .calculus import defun
def update_psum(self, S):
    """
        This routine applies the convergence acceleration to the list of partial sums.

        A   = sum(a_k, k = 0..infinity)
        s_n = sum(a_k ,k = 0..n)

        v, e = ...update_psum([s_0, s_1,..., s_k])

        output:
          v      current estimate of the series A
          e      an error estimate which is simply the difference between the current
                 estimate and the last estimate.
        """
    n = len(S)
    d = (3 + self.ctx.sqrt(8)) ** n
    d = (d + 1 / d) / 2
    b = self.ctx.one
    s = 0
    for k in xrange(n):
        b = 2 * (n + k) * (n - k) * b / ((2 * k + 1) * (k + self.ctx.one))
        s += b * S[k]
    value = s / d
    err = abs(value - self.last)
    self.last = value
    return (value, err)