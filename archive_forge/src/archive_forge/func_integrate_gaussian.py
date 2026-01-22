import warnings
from scipy import linalg, special
from scipy._lib._util import check_random_state
from numpy import (asarray, atleast_2d, reshape, zeros, newaxis, exp, pi,
import numpy as np
from . import _mvn
from ._stats import gaussian_kernel_estimate, gaussian_kernel_estimate_log
from scipy.special import logsumexp  # noqa: F401
def integrate_gaussian(self, mean, cov):
    """
        Multiply estimated density by a multivariate Gaussian and integrate
        over the whole space.

        Parameters
        ----------
        mean : aray_like
            A 1-D array, specifying the mean of the Gaussian.
        cov : array_like
            A 2-D array, specifying the covariance matrix of the Gaussian.

        Returns
        -------
        result : scalar
            The value of the integral.

        Raises
        ------
        ValueError
            If the mean or covariance of the input Gaussian differs from
            the KDE's dimensionality.

        """
    mean = atleast_1d(squeeze(mean))
    cov = atleast_2d(cov)
    if mean.shape != (self.d,):
        raise ValueError('mean does not have dimension %s' % self.d)
    if cov.shape != (self.d, self.d):
        raise ValueError('covariance does not have dimension %s' % self.d)
    mean = mean[:, newaxis]
    sum_cov = self.covariance + cov
    sum_cov_chol = linalg.cho_factor(sum_cov)
    diff = self.dataset - mean
    tdiff = linalg.cho_solve(sum_cov_chol, diff)
    sqrt_det = np.prod(np.diagonal(sum_cov_chol[0]))
    norm_const = power(2 * pi, sum_cov.shape[0] / 2.0) * sqrt_det
    energies = sum(diff * tdiff, axis=0) / 2.0
    result = sum(exp(-energies) * self.weights, axis=0) / norm_const
    return result