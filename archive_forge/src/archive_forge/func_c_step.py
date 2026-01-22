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
def c_step(X, n_support, remaining_iterations=30, initial_estimates=None, verbose=False, cov_computation_method=empirical_covariance, random_state=None):
    """C_step procedure described in [Rouseeuw1984]_ aiming at computing MCD.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data set in which we look for the n_support observations whose
        scatter matrix has minimum determinant.

    n_support : int
        Number of observations to compute the robust estimates of location
        and covariance from. This parameter must be greater than
        `n_samples / 2`.

    remaining_iterations : int, default=30
        Number of iterations to perform.
        According to [Rouseeuw1999]_, two iterations are sufficient to get
        close to the minimum, and we never need more than 30 to reach
        convergence.

    initial_estimates : tuple of shape (2,), default=None
        Initial estimates of location and shape from which to run the c_step
        procedure:
        - initial_estimates[0]: an initial location estimate
        - initial_estimates[1]: an initial covariance estimate

    verbose : bool, default=False
        Verbose mode.

    cov_computation_method : callable,             default=:func:`sklearn.covariance.empirical_covariance`
        The function which will be used to compute the covariance.
        Must return array of shape (n_features, n_features).

    random_state : int, RandomState instance or None, default=None
        Determines the pseudo random number generator for shuffling the data.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    location : ndarray of shape (n_features,)
        Robust location estimates.

    covariance : ndarray of shape (n_features, n_features)
        Robust covariance estimates.

    support : ndarray of shape (n_samples,)
        A mask for the `n_support` observations whose scatter matrix has
        minimum determinant.

    References
    ----------
    .. [Rouseeuw1999] A Fast Algorithm for the Minimum Covariance Determinant
        Estimator, 1999, American Statistical Association and the American
        Society for Quality, TECHNOMETRICS
    """
    X = np.asarray(X)
    random_state = check_random_state(random_state)
    return _c_step(X, n_support, remaining_iterations=remaining_iterations, initial_estimates=initial_estimates, verbose=verbose, cov_computation_method=cov_computation_method, random_state=random_state)