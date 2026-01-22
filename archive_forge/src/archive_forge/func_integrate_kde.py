import warnings
from scipy import linalg, special
from scipy._lib._util import check_random_state
from numpy import (asarray, atleast_2d, reshape, zeros, newaxis, exp, pi,
import numpy as np
from . import _mvn
from ._stats import gaussian_kernel_estimate, gaussian_kernel_estimate_log
from scipy.special import logsumexp  # noqa: F401
def integrate_kde(self, other):
    """
        Computes the integral of the product of this  kernel density estimate
        with another.

        Parameters
        ----------
        other : gaussian_kde instance
            The other kde.

        Returns
        -------
        value : scalar
            The result of the integral.

        Raises
        ------
        ValueError
            If the KDEs have different dimensionality.

        """
    if other.d != self.d:
        raise ValueError('KDEs are not the same dimensionality')
    if other.n < self.n:
        small = other
        large = self
    else:
        small = self
        large = other
    sum_cov = small.covariance + large.covariance
    sum_cov_chol = linalg.cho_factor(sum_cov)
    result = 0.0
    for i in range(small.n):
        mean = small.dataset[:, i, newaxis]
        diff = large.dataset - mean
        tdiff = linalg.cho_solve(sum_cov_chol, diff)
        energies = sum(diff * tdiff, axis=0) / 2.0
        result += sum(exp(-energies) * large.weights, axis=0) * small.weights[i]
    sqrt_det = np.prod(np.diagonal(sum_cov_chol[0]))
    norm_const = power(2 * pi, sum_cov.shape[0] / 2.0) * sqrt_det
    result /= norm_const
    return result