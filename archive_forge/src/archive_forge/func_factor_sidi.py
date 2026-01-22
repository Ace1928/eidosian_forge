from ..libmp.backend import xrange
from .calculus import defun
def factor_sidi(self, i):
    return (self.theta + self.n - 1) * (self.theta + self.n - 2) / self.ctx.mpf((self.theta + 2 * self.n - i - 2) * (self.theta + 2 * self.n - i - 3))