import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
def ttost_paired(x1, x2, low, upp, transform=None, weights=None):
    """test of (non-)equivalence for two dependent, paired sample

    TOST: two one-sided t tests

    null hypothesis:  md < low or md > upp
    alternative hypothesis:  low < md < upp

    where md is the mean, expected value of the difference x1 - x2

    If the pvalue is smaller than a threshold,say 0.05, then we reject the
    hypothesis that the difference between the two samples is larger than the
    the thresholds given by low and upp.

    Parameters
    ----------
    x1 : array_like
        first of the two independent samples
    x2 : array_like
        second of the two independent samples
    low, upp : float
        equivalence interval low < mean of difference < upp
    weights : None or ndarray
        case weights for the two samples. For details on weights see
        ``DescrStatsW``
    transform : None or function
        If None (default), then the data is not transformed. Given a function
        sample data and thresholds are transformed. If transform is log the
        the equivalence interval is in ratio: low < x1 / x2 < upp

    Returns
    -------
    pvalue : float
        pvalue of the non-equivalence test
    t1, pv1, df1 : tuple
        test statistic, pvalue and degrees of freedom for lower threshold test
    t2, pv2, df2 : tuple
        test statistic, pvalue and degrees of freedom for upper threshold test

    """
    if transform:
        if transform is np.log:
            x1 = transform(x1)
            x2 = transform(x2)
        else:
            xx = transform(np.concatenate((x1, x2), 0))
            x1 = xx[:len(x1)]
            x2 = xx[len(x1):]
        low = transform(low)
        upp = transform(upp)
    dd = DescrStatsW(x1 - x2, weights=weights, ddof=0)
    t1, pv1, df1 = dd.ttest_mean(low, alternative='larger')
    t2, pv2, df2 = dd.ttest_mean(upp, alternative='smaller')
    return (np.maximum(pv1, pv2), (t1, pv1, df1), (t2, pv2, df2))