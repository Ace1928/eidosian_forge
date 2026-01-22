import warnings
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from scipy.stats import chi2
from ..base import _fit_context
from ..utils import check_array, check_random_state
from ..utils._param_validation import Interval
from ..utils.extmath import fast_logdet
from ._empirical_covariance import EmpiricalCovariance, empirical_covariance
def correct_covariance(self, data):
    """Apply a correction to raw Minimum Covariance Determinant estimates.

        Correction using the empirical correction factor suggested
        by Rousseeuw and Van Driessen in [RVD]_.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The data matrix, with p features and n samples.
            The data set must be the one which was used to compute
            the raw estimates.

        Returns
        -------
        covariance_corrected : ndarray of shape (n_features, n_features)
            Corrected robust covariance estimate.

        References
        ----------

        .. [RVD] A Fast Algorithm for the Minimum Covariance
            Determinant Estimator, 1999, American Statistical Association
            and the American Society for Quality, TECHNOMETRICS
        """
    n_samples = len(self.dist_)
    n_support = np.sum(self.support_)
    if n_support < n_samples and np.allclose(self.raw_covariance_, 0):
        raise ValueError('The covariance matrix of the support data is equal to 0, try to increase support_fraction')
    correction = np.median(self.dist_) / chi2(data.shape[1]).isf(0.5)
    covariance_corrected = self.raw_covariance_ * correction
    self.dist_ /= correction
    return covariance_corrected