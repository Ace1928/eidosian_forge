import numpy as np
from numpy import ndarray
import numpy.ma as ma
from numpy.ma import masked, nomask
import math
import itertools
import warnings
from collections import namedtuple
from . import distributions
from scipy._lib._util import _rename_parameter, _contains_nan
from scipy._lib._bunch import _make_tuple_bunch
import scipy.special as special
import scipy.stats._stats_py
from ._stats_mstats_common import (
def sen_seasonal_slopes(x):
    """
    Computes seasonal Theil-Sen and Kendall slope estimators.

    The seasonal generalization of Sen's slope computes the slopes between all
    pairs of values within a "season" (column) of a 2D array. It returns an
    array containing the median of these "within-season" slopes for each
    season (the Theil-Sen slope estimator of each season), and it returns the
    median of the within-season slopes across all seasons (the seasonal Kendall
    slope estimator).

    Parameters
    ----------
    x : 2D array_like
        Each column of `x` contains measurements of the dependent variable
        within a season. The independent variable (usually time) of each season
        is assumed to be ``np.arange(x.shape[0])``.

    Returns
    -------
    result : ``SenSeasonalSlopesResult`` instance
        The return value is an object with the following attributes:

        intra_slope : ndarray
            For each season, the Theil-Sen slope estimator: the median of
            within-season slopes.
        inter_slope : float
            The seasonal Kendall slope estimateor: the median of within-season
            slopes *across all* seasons.

    See Also
    --------
    theilslopes : the analogous function for non-seasonal data
    scipy.stats.theilslopes : non-seasonal slopes for non-masked arrays

    Notes
    -----
    The slopes :math:`d_{ijk}` within season :math:`i` are:

    .. math::

        d_{ijk} = \\frac{x_{ij} - x_{ik}}
                            {j - k}

    for pairs of distinct integer indices :math:`j, k` of :math:`x`.

    Element :math:`i` of the returned `intra_slope` array is the median of the
    :math:`d_{ijk}` over all :math:`j < k`; this is the Theil-Sen slope
    estimator of season :math:`i`. The returned `inter_slope` value, better
    known as the seasonal Kendall slope estimator, is the median of the
    :math:`d_{ijk}` over all :math:`i, j, k`.

    References
    ----------
    .. [1] Hirsch, Robert M., James R. Slack, and Richard A. Smith.
           "Techniques of trend analysis for monthly water quality data."
           *Water Resources Research* 18.1 (1982): 107-121.

    Examples
    --------
    Suppose we have 100 observations of a dependent variable for each of four
    seasons:

    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> x = rng.random(size=(100, 4))

    We compute the seasonal slopes as:

    >>> from scipy import stats
    >>> intra_slope, inter_slope = stats.mstats.sen_seasonal_slopes(x)

    If we define a function to compute all slopes between observations within
    a season:

    >>> def dijk(yi):
    ...     n = len(yi)
    ...     x = np.arange(n)
    ...     dy = yi - yi[:, np.newaxis]
    ...     dx = x - x[:, np.newaxis]
    ...     # we only want unique pairs of distinct indices
    ...     mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    ...     return dy[mask]/dx[mask]

    then element ``i`` of ``intra_slope`` is the median of ``dijk[x[:, i]]``:

    >>> i = 2
    >>> np.allclose(np.median(dijk(x[:, i])), intra_slope[i])
    True

    and ``inter_slope`` is the median of the values returned by ``dijk`` for
    all seasons:

    >>> all_slopes = np.concatenate([dijk(x[:, i]) for i in range(x.shape[1])])
    >>> np.allclose(np.median(all_slopes), inter_slope)
    True

    Because the data are randomly generated, we would expect the median slopes
    to be nearly zero both within and across all seasons, and indeed they are:

    >>> intra_slope.data
    array([ 0.00124504, -0.00277761, -0.00221245, -0.00036338])
    >>> inter_slope
    -0.0010511779872922058

    """
    x = ma.array(x, subok=True, copy=False, ndmin=2)
    n, _ = x.shape
    szn_slopes = ma.vstack([(x[i + 1:] - x[i]) / np.arange(1, n - i)[:, None] for i in range(n)])
    szn_medslopes = ma.median(szn_slopes, axis=0)
    medslope = ma.median(szn_slopes, axis=None)
    return SenSeasonalSlopesResult(szn_medslopes, medslope)