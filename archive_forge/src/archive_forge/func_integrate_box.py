import warnings
from scipy import linalg, special
from scipy._lib._util import check_random_state
from numpy import (asarray, atleast_2d, reshape, zeros, newaxis, exp, pi,
import numpy as np
from . import _mvn
from ._stats import gaussian_kernel_estimate, gaussian_kernel_estimate_log
from scipy.special import logsumexp  # noqa: F401
def integrate_box(self, low_bounds, high_bounds, maxpts=None):
    """Computes the integral of a pdf over a rectangular interval.

        Parameters
        ----------
        low_bounds : array_like
            A 1-D array containing the lower bounds of integration.
        high_bounds : array_like
            A 1-D array containing the upper bounds of integration.
        maxpts : int, optional
            The maximum number of points to use for integration.

        Returns
        -------
        value : scalar
            The result of the integral.

        """
    if maxpts is not None:
        extra_kwds = {'maxpts': maxpts}
    else:
        extra_kwds = {}
    value, inform = _mvn.mvnun_weighted(low_bounds, high_bounds, self.dataset, self.weights, self.covariance, **extra_kwds)
    if inform:
        msg = 'An integral in _mvn.mvnun requires more points than %s' % (self.d * 1000)
        warnings.warn(msg, stacklevel=2)
    return value