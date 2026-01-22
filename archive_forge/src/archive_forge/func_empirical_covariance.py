import warnings
import numpy as np
from scipy import linalg
from .. import config_context
from ..base import BaseEstimator, _fit_context
from ..metrics.pairwise import pairwise_distances
from ..utils import check_array
from ..utils._param_validation import validate_params
from ..utils.extmath import fast_logdet
@validate_params({'X': ['array-like'], 'assume_centered': ['boolean']}, prefer_skip_nested_validation=True)
def empirical_covariance(X, *, assume_centered=False):
    """Compute the Maximum likelihood covariance estimator.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data from which to compute the covariance estimate.

    assume_centered : bool, default=False
        If `True`, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If `False`, data will be centered before computation.

    Returns
    -------
    covariance : ndarray of shape (n_features, n_features)
        Empirical covariance (Maximum Likelihood Estimator).

    Examples
    --------
    >>> from sklearn.covariance import empirical_covariance
    >>> X = [[1,1,1],[1,1,1],[1,1,1],
    ...      [0,0,0],[0,0,0],[0,0,0]]
    >>> empirical_covariance(X)
    array([[0.25, 0.25, 0.25],
           [0.25, 0.25, 0.25],
           [0.25, 0.25, 0.25]])
    """
    X = check_array(X, ensure_2d=False, force_all_finite=False)
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))
    if X.shape[0] == 1:
        warnings.warn('Only one sample available. You may want to reshape your data array')
    if assume_centered:
        covariance = np.dot(X.T, X) / X.shape[0]
    else:
        covariance = np.cov(X.T, bias=1)
    if covariance.ndim == 0:
        covariance = np.array([[covariance]])
    return covariance