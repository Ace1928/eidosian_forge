import numpy as np
from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import (
def cohensd2problarger(d):
    """
    Convert Cohen's d effect size to stochastically-larger-probability.

    This assumes observations are normally distributed.

    Computed as

        p = Prob(x1 > x2) = F(d / sqrt(2))

    where `F` is cdf of normal distribution. Cohen's d is defined as

        d = (mean1 - mean2) / std

    where ``std`` is the pooled within standard deviation.

    Parameters
    ----------
    d : float or array_like
        Cohen's d effect size for difference mean1 - mean2.

    Returns
    -------
    prob : float or ndarray
        Prob(x1 > x2)
    """
    return stats.norm.cdf(d / np.sqrt(2))