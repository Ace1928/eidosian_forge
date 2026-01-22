from collections import namedtuple
import warnings
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
def distance_correlation(x, y):
    """Distance correlation.

    Calculate the empirical distance correlation as described in [1]_.
    This statistic is analogous to product-moment correlation and describes
    the dependence between `x` and `y`, which are random vectors of
    arbitrary length. The statistics' values range between 0 (implies
    independence) and 1 (implies complete dependence).

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

    Returns
    -------
    float
        The empirical distance correlation between `x` and `y`.

    References
    ----------
    .. [1] Szekely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007)
       "Measuring and testing dependence by correlation of distances".
       Annals of Statistics, Vol. 35 No. 6, pp. 2769-2794.

    Examples
    --------

    >>> from statsmodels.stats.dist_dependence_measures import
    ... distance_correlation
    >>> distance_correlation(np.random.random(1000), np.random.random(1000))
    0.04060497840149489

    """
    return distance_statistics(x, y).distance_correlation