import warnings
from scipy import linalg, special
from scipy._lib._util import check_random_state
from numpy import (asarray, atleast_2d, reshape, zeros, newaxis, exp, pi,
import numpy as np
from . import _mvn
from ._stats import gaussian_kernel_estimate, gaussian_kernel_estimate_log
from scipy.special import logsumexp  # noqa: F401
def integrate_box_1d(self, low, high):
    """
        Computes the integral of a 1D pdf between two bounds.

        Parameters
        ----------
        low : scalar
            Lower bound of integration.
        high : scalar
            Upper bound of integration.

        Returns
        -------
        value : scalar
            The result of the integral.

        Raises
        ------
        ValueError
            If the KDE is over more than one dimension.

        """
    if self.d != 1:
        raise ValueError('integrate_box_1d() only handles 1D pdfs')
    stdev = ravel(sqrt(self.covariance))[0]
    normalized_low = ravel((low - self.dataset) / stdev)
    normalized_high = ravel((high - self.dataset) / stdev)
    value = np.sum(self.weights * (special.ndtr(normalized_high) - special.ndtr(normalized_low)))
    return value