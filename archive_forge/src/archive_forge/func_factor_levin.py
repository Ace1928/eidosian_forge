from ..libmp.backend import xrange
from .calculus import defun
def factor_levin(self, i):
    return (self.theta + i) * (self.theta + self.n - 1) ** (self.n - i - 2) / self.ctx.mpf(self.theta + self.n) ** (self.n - i - 1)