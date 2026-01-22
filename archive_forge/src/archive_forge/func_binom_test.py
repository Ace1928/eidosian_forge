from statsmodels.compat.python import lzip
from typing import Callable
import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.base import AllPairsResults, HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.tools.validation import array_like
def binom_test(count, nobs, prop=0.5, alternative='two-sided'):
    """
    Perform a test that the probability of success is p.

    This is an exact, two-sided test of the null hypothesis
    that the probability of success in a Bernoulli experiment
    is `p`.

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials.
    nobs : int
        the number of trials or observations.
    prop : float, optional
        The probability of success under the null hypothesis,
        `0 <= prop <= 1`. The default value is `prop = 0.5`
    alternative : str in ['two-sided', 'smaller', 'larger']
        alternative hypothesis, which can be two-sided or either one of the
        one-sided tests.

    Returns
    -------
    p-value : float
        The p-value of the hypothesis test

    Notes
    -----
    This uses scipy.stats.binom_test for the two-sided alternative.
    """
    if np.any(prop > 1.0) or np.any(prop < 0.0):
        raise ValueError('p must be in range [0,1]')
    if alternative in ['2s', 'two-sided']:
        try:
            pval = stats.binomtest(count, n=nobs, p=prop).pvalue
        except AttributeError:
            pval = stats.binom_test(count, n=nobs, p=prop)
    elif alternative in ['l', 'larger']:
        pval = stats.binom.sf(count - 1, nobs, prop)
    elif alternative in ['s', 'smaller']:
        pval = stats.binom.cdf(count, nobs, prop)
    else:
        raise ValueError('alternative not recognized\nshould be two-sided, larger or smaller')
    return pval