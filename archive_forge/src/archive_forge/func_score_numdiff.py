import numpy as np
from ._penalties import NonePenalty
from statsmodels.tools.numdiff import approx_fprime_cs, approx_fprime
def score_numdiff(self, params, pen_weight=None, method='fd', **kwds):
    """score based on finite difference derivative
        """
    if pen_weight is None:
        pen_weight = self.pen_weight
    loglike = lambda p: self.loglike(p, pen_weight=pen_weight, **kwds)
    if method == 'cs':
        return approx_fprime_cs(params, loglike)
    elif method == 'fd':
        return approx_fprime(params, loglike, centered=True)
    else:
        raise ValueError('method not recognized, should be "fd" or "cs"')