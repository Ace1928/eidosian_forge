from statsmodels.compat.python import lzip, lfilter
import numpy as np
import scipy.integrate
from scipy.special import factorial
from numpy import exp, multiply, square, divide, subtract, inf
def smoothvar(self, xs, ys, x):
    """
        Returns the kernel smoothing estimate of the variance at point x.
        """
    xs, ys = self.in_domain(xs, ys, x)
    if len(xs) > 0:
        fittedvals = np.array([self.smooth(xs, ys, xx) for xx in xs])
        rs = square(subtract(ys, fittedvals))
        w = np.sum(square(subtract(1.0, square(divide(subtract(xs, x), self.h)))))
        v = np.sum(multiply(rs, square(subtract(1, square(divide(subtract(xs, x), self.h))))))
        return v / w
    else:
        return np.nan