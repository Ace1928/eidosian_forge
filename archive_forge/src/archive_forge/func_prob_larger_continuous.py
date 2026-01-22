import numpy as np
from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import (
def prob_larger_continuous(distr1, distr2):
    """
    Probability indicating that distr1 is stochastically larger than distr2.

    This computes

        p = P(x1 > x2)

    for two continuous distributions, where `distr1` and `distr2` are the
    distributions of random variables x1 and x2 respectively.

    Parameters
    ----------
    distr1, distr2 : distributions
        Two instances of scipy.stats.distributions. The required methods are
        cdf of the second distribution and expect of the first distribution.

    Returns
    -------
    p : probability x1 is larger than x2


    Notes
    -----
    This is a one-liner that is added mainly as reference.

    Examples
    --------
    >>> from scipy import stats
    >>> prob_larger_continuous(stats.norm, stats.t(5))
    0.4999999999999999

    # which is the same as
    >>> stats.norm.expect(stats.t(5).cdf)
    0.4999999999999999

    # distribution 1 with smaller mean (loc) than distribution 2
    >>> prob_larger_continuous(stats.norm, stats.norm(loc=1))
    0.23975006109347669

    """
    return distr1.expect(distr2.cdf)