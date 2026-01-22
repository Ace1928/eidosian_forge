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
def binom_tost(count, nobs, low, upp):
    """
    Exact TOST test for one proportion using binomial distribution

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials.
    nobs : int
        the number of trials or observations.
    low, upp : floats
        lower and upper limit of equivalence region

    Returns
    -------
    pvalue : float
        p-value of equivalence test
    pval_low, pval_upp : floats
        p-values of lower and upper one-sided tests

    """
    tt1 = binom_test(count, nobs, alternative='larger', prop=low)
    tt2 = binom_test(count, nobs, alternative='smaller', prop=upp)
    return (np.maximum(tt1, tt2), tt1, tt2)