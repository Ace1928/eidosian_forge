import warnings
from numbers import Integral, Real
import numpy as np
from ..base import _fit_context
from ..utils import check_array
from ..utils._param_validation import Interval, validate_params
from . import EmpiricalCovariance, empirical_covariance
@validate_params({'emp_cov': ['array-like'], 'shrinkage': [Interval(Real, 0, 1, closed='both')]}, prefer_skip_nested_validation=True)
def shrunk_covariance(emp_cov, shrinkage=0.1):
    """Calculate covariance matrices shrunk on the diagonal.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    emp_cov : array-like of shape (..., n_features, n_features)
        Covariance matrices to be shrunk, at least 2D ndarray.

    shrinkage : float, default=0.1
        Coefficient in the convex combination used for the computation
        of the shrunk estimate. Range is [0, 1].

    Returns
    -------
    shrunk_cov : ndarray of shape (..., n_features, n_features)
        Shrunk covariance matrices.

    Notes
    -----
    The regularized (shrunk) covariance is given by::

        (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where `mu = trace(cov) / n_features`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_gaussian_quantiles
    >>> from sklearn.covariance import empirical_covariance, shrunk_covariance
    >>> real_cov = np.array([[.8, .3], [.3, .4]])
    >>> rng = np.random.RandomState(0)
    >>> X = rng.multivariate_normal(mean=[0, 0], cov=real_cov, size=500)
    >>> shrunk_covariance(empirical_covariance(X))
    array([[0.73..., 0.25...],
           [0.25..., 0.41...]])
    """
    emp_cov = check_array(emp_cov, allow_nd=True)
    n_features = emp_cov.shape[-1]
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    mu = np.trace(emp_cov, axis1=-2, axis2=-1) / n_features
    mu = np.expand_dims(mu, axis=tuple(range(mu.ndim, emp_cov.ndim)))
    shrunk_cov += shrinkage * mu * np.eye(n_features)
    return shrunk_cov