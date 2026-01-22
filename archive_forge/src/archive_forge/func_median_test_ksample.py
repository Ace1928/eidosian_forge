import numpy as np
from scipy import stats
from scipy.special import comb
import warnings
from statsmodels.tools.validation import array_like
def median_test_ksample(x, groups):
    """chisquare test for equality of median/location

    This tests whether all groups have the same fraction of observations
    above the median.

    Parameters
    ----------
    x : array_like
        data values stacked for all groups
    groups : array_like
        group labels or indicator

    Returns
    -------
    stat : float
       test statistic
    pvalue : float
       pvalue from the chisquare distribution
    others ????
       currently some test output, table and expected

    """
    x = np.asarray(x)
    gruni = np.unique(groups)
    xli = [x[groups == group] for group in gruni]
    xmedian = np.median(x)
    counts_larger = np.array([(xg > xmedian).sum() for xg in xli])
    counts = np.array([len(xg) for xg in xli])
    counts_smaller = counts - counts_larger
    nobs = counts.sum()
    n_larger = (x > xmedian).sum()
    n_smaller = nobs - n_larger
    table = np.vstack((counts_smaller, counts_larger))
    expected = np.vstack((counts * 1.0 / nobs * n_smaller, counts * 1.0 / nobs * n_larger))
    if (expected < 5).any():
        print('Warning: There are cells with less than 5 expectedobservations. The chisquare distribution might not be a goodapproximation for the true distribution.')
    return (stats.chisquare(table.ravel(), expected.ravel(), ddof=1), table, expected)