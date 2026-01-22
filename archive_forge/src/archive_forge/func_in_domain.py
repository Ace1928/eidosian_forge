from statsmodels.compat.python import lzip, lfilter
import numpy as np
import scipy.integrate
from scipy.special import factorial
from numpy import exp, multiply, square, divide, subtract, inf
def in_domain(self, xs, ys, x):
    """
        Returns the filtered (xs, ys) based on the Kernel domain centred on x
        """

    def isInDomain(xy):
        """Used for filter to check if point is in the domain"""
        u = (xy[0] - x) / self.h
        return np.all((u >= self.domain[0]) & (u <= self.domain[1]))
    if self.domain is None:
        return (xs, ys)
    else:
        filtered = lfilter(isInDomain, lzip(xs, ys))
        if len(filtered) > 0:
            xs, ys = lzip(*filtered)
            return (xs, ys)
        else:
            return ([], [])