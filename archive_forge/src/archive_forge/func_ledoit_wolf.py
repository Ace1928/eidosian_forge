import warnings
from numbers import Integral, Real
import numpy as np
from ..base import _fit_context
from ..utils import check_array
from ..utils._param_validation import Interval, validate_params
from . import EmpiricalCovariance, empirical_covariance
@validate_params({'X': ['array-like']}, prefer_skip_nested_validation=False)
def ledoit_wolf(X, *, assume_centered=False, block_size=1000):
    """Estimate the shrunk Ledoit-Wolf covariance matrix.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data from which to compute the covariance estimate.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, data will be centered before computation.

    block_size : int, default=1000
        Size of blocks into which the covariance matrix will be split.
        This is purely a memory optimization and does not affect results.

    Returns
    -------
    shrunk_cov : ndarray of shape (n_features, n_features)
        Shrunk covariance.

    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Notes
    -----
    The regularized (shrunk) covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import empirical_covariance, ledoit_wolf
    >>> real_cov = np.array([[.4, .2], [.2, .8]])
    >>> rng = np.random.RandomState(0)
    >>> X = rng.multivariate_normal(mean=[0, 0], cov=real_cov, size=50)
    >>> covariance, shrinkage = ledoit_wolf(X)
    >>> covariance
    array([[0.44..., 0.16...],
           [0.16..., 0.80...]])
    >>> shrinkage
    0.23...
    """
    estimator = LedoitWolf(assume_centered=assume_centered, block_size=block_size, store_precision=False).fit(X)
    return (estimator.covariance_, estimator.shrinkage_)