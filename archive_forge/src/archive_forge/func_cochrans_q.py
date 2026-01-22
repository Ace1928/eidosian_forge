import numpy as np
from scipy import stats
from scipy.special import comb
import warnings
from statsmodels.tools.validation import array_like
def cochrans_q(x):
    """Cochran's Q test for identical effect of k treatments

    Cochran's Q is a k-sample extension of the McNemar test. If there are only
    two treatments, then Cochran's Q test and McNemar test are equivalent.

    Test that the probability of success is the same for each treatment.
    The alternative is that at least two treatments have a different
    probability of success.

    Parameters
    ----------
    x : array_like, 2d (N,k)
        data with N cases and k variables

    Returns
    -------
    q_stat : float
       test statistic
    pvalue : float
       pvalue from the chisquare distribution

    Notes
    -----
    In Wikipedia terminology, rows are blocks and columns are treatments.
    The number of rows N, should be large for the chisquare distribution to be
    a good approximation.
    The Null hypothesis of the test is that all treatments have the
    same effect.

    References
    ----------
    https://en.wikipedia.org/wiki/Cochran_test
    SAS Manual for NPAR TESTS

    """
    warnings.warn('Deprecated, use stats.cochrans_q instead', FutureWarning)
    x = np.asarray(x)
    gruni = np.unique(x)
    N, k = x.shape
    count_row_success = (x == gruni[-1]).sum(1, float)
    count_col_success = (x == gruni[-1]).sum(0, float)
    count_row_ss = count_row_success.sum()
    count_col_ss = count_col_success.sum()
    assert count_row_ss == count_col_ss
    q_stat = (k - 1) * (k * np.sum(count_col_success ** 2) - count_col_ss ** 2) / (k * count_row_ss - np.sum(count_row_success ** 2))
    return (q_stat, stats.chi2.sf(q_stat, k - 1))