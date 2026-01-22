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
def select_candidates(X, n_support, n_trials, select=1, n_iter=30, verbose=False, cov_computation_method=empirical_covariance, random_state=None):
    """Finds the best pure subset of observations to compute MCD from it.

    The purpose of this function is to find the best sets of n_support
    observations with respect to a minimization of their covariance
    matrix determinant. Equivalently, it removes n_samples-n_support
    observations to construct what we call a pure data set (i.e. not
    containing outliers). The list of the observations of the pure
    data set is referred to as the `support`.

    Starting from a random support, the pure data set is found by the
    c_step procedure introduced by Rousseeuw and Van Driessen in
    [RV]_.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data (sub)set in which we look for the n_support purest observations.

    n_support : int
        The number of samples the pure data set must contain.
        This parameter must be in the range `[(n + p + 1)/2] < n_support < n`.

    n_trials : int or tuple of shape (2,)
        Number of different initial sets of observations from which to
        run the algorithm. This parameter should be a strictly positive
        integer.
        Instead of giving a number of trials to perform, one can provide a
        list of initial estimates that will be used to iteratively run
        c_step procedures. In this case:
        - n_trials[0]: array-like, shape (n_trials, n_features)
          is the list of `n_trials` initial location estimates
        - n_trials[1]: array-like, shape (n_trials, n_features, n_features)
          is the list of `n_trials` initial covariances estimates

    select : int, default=1
        Number of best candidates results to return. This parameter must be
        a strictly positive integer.

    n_iter : int, default=30
        Maximum number of iterations for the c_step procedure.
        (2 is enough to be close to the final solution. "Never" exceeds 20).
        This parameter must be a strictly positive integer.

    verbose : bool, default=False
        Control the output verbosity.

    cov_computation_method : callable,             default=:func:`sklearn.covariance.empirical_covariance`
        The function which will be used to compute the covariance.
        Must return an array of shape (n_features, n_features).

    random_state : int, RandomState instance or None, default=None
        Determines the pseudo random number generator for shuffling the data.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    See Also
    ---------
    c_step

    Returns
    -------
    best_locations : ndarray of shape (select, n_features)
        The `select` location estimates computed from the `select` best
        supports found in the data set (`X`).

    best_covariances : ndarray of shape (select, n_features, n_features)
        The `select` covariance estimates computed from the `select`
        best supports found in the data set (`X`).

    best_supports : ndarray of shape (select, n_samples)
        The `select` best supports found in the data set (`X`).

    References
    ----------
    .. [RV] A Fast Algorithm for the Minimum Covariance Determinant
        Estimator, 1999, American Statistical Association and the American
        Society for Quality, TECHNOMETRICS
    """
    random_state = check_random_state(random_state)
    if isinstance(n_trials, Integral):
        run_from_estimates = False
    elif isinstance(n_trials, tuple):
        run_from_estimates = True
        estimates_list = n_trials
        n_trials = estimates_list[0].shape[0]
    else:
        raise TypeError("Invalid 'n_trials' parameter, expected tuple or  integer, got %s (%s)" % (n_trials, type(n_trials)))
    all_estimates = []
    if not run_from_estimates:
        for j in range(n_trials):
            all_estimates.append(_c_step(X, n_support, remaining_iterations=n_iter, verbose=verbose, cov_computation_method=cov_computation_method, random_state=random_state))
    else:
        for j in range(n_trials):
            initial_estimates = (estimates_list[0][j], estimates_list[1][j])
            all_estimates.append(_c_step(X, n_support, remaining_iterations=n_iter, initial_estimates=initial_estimates, verbose=verbose, cov_computation_method=cov_computation_method, random_state=random_state))
    all_locs_sub, all_covs_sub, all_dets_sub, all_supports_sub, all_ds_sub = zip(*all_estimates)
    index_best = np.argsort(all_dets_sub)[:select]
    best_locations = np.asarray(all_locs_sub)[index_best]
    best_covariances = np.asarray(all_covs_sub)[index_best]
    best_supports = np.asarray(all_supports_sub)[index_best]
    best_ds = np.asarray(all_ds_sub)[index_best]
    return (best_locations, best_covariances, best_supports, best_ds)