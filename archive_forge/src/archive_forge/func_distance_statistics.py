from collections import namedtuple
import warnings
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
def distance_statistics(x, y, x_dist=None, y_dist=None):
    """Calculate various distance dependence statistics.

    Calculate several distance dependence statistics as described in [1]_.

    Parameters
    ----------
    x : array_like, 1-D or 2-D
        If `x` is 1-D than it is assumed to be a vector of observations of a
        single random variable. If `x` is 2-D than the rows should be
        observations and the columns are treated as the components of a
        random vector, i.e., each column represents a different component of
        the random vector `x`.
    y : array_like, 1-D or 2-D
        Same as `x`, but only the number of observation has to match that of
        `x`. If `y` is 2-D note that the number of columns of `y` (i.e., the
        number of components in the random vector) does not need to match
        the number of columns in `x`.
    x_dist : array_like, 2-D, optional
        A square 2-D array_like object whose values are the euclidean
        distances between `x`'s rows.
    y_dist : array_like, 2-D, optional
        A square 2-D array_like object whose values are the euclidean
        distances between `y`'s rows.

    Returns
    -------
    namedtuple
        A named tuple of distance dependence statistics (DistDependStat) with
        the following values:

        - test_statistic : float - The "basic" test statistic (i.e., the one
          used when the `emp` method is chosen when calling
          ``distance_covariance_test()``
        - distance_correlation : float - The distance correlation
          between `x` and `y`.
        - distance_covariance : float - The distance covariance of
          `x` and `y`.
        - dvar_x : float - The distance variance of `x`.
        - dvar_y : float - The distance variance of `y`.
        - S : float - The mean of the euclidean distances in `x` multiplied
          by those of `y`. Mostly used internally.

    References
    ----------
    .. [1] Szekely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007)
       "Measuring and testing dependence by correlation of distances".
       Annals of Statistics, Vol. 35 No. 6, pp. 2769-2794.

    Examples
    --------

    >>> from statsmodels.stats.dist_dependence_measures import
    ... distance_statistics
    >>> distance_statistics(np.random.random(1000), np.random.random(1000))
    DistDependStat(test_statistic=0.07948284320205831,
    distance_correlation=0.04269511890990793,
    distance_covariance=0.008915315092696293,
    dvar_x=0.20719027438266704, dvar_y=0.21044934264957588,
    S=0.10892061635588891)

    """
    x, y = _validate_and_tranform_x_and_y(x, y)
    n = x.shape[0]
    a = x_dist if x_dist is not None else squareform(pdist(x, 'euclidean'))
    b = y_dist if y_dist is not None else squareform(pdist(y, 'euclidean'))
    a_row_means = a.mean(axis=0, keepdims=True)
    b_row_means = b.mean(axis=0, keepdims=True)
    a_col_means = a.mean(axis=1, keepdims=True)
    b_col_means = b.mean(axis=1, keepdims=True)
    a_mean = a.mean()
    b_mean = b.mean()
    A = a - a_row_means - a_col_means + a_mean
    B = b - b_row_means - b_col_means + b_mean
    S = a_mean * b_mean
    dcov = np.sqrt(np.multiply(A, B).mean())
    dvar_x = np.sqrt(np.multiply(A, A).mean())
    dvar_y = np.sqrt(np.multiply(B, B).mean())
    dcor = dcov / np.sqrt(dvar_x * dvar_y)
    test_statistic = n * dcov ** 2
    return DistDependStat(test_statistic=test_statistic, distance_correlation=dcor, distance_covariance=dcov, dvar_x=dvar_x, dvar_y=dvar_y, S=S)