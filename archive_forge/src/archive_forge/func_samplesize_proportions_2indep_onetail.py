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
def samplesize_proportions_2indep_onetail(diff, prop2, power, ratio=1, alpha=0.05, value=0, alternative='two-sided'):
    """
    Required sample size assuming normal distribution based on one tail

    This uses an explicit computation for the sample size that is required
    to achieve a given power corresponding to the appropriate tails of the
    normal distribution. This ignores the far tail in a two-sided test
    which is negligible in the common case when alternative and null are
    far apart.

    Parameters
    ----------
    diff : float
        Difference between proportion 1 and 2 under the alternative
    prop2 : float
        proportion for the reference case, prop2, proportions for the
        first case will be computing using p2 and diff
        p1 = p2 + diff
    power : float
        Power for which sample size is computed.
    ratio : float
        Sample size ratio, nobs2 = ratio * nobs1
    alpha : float in interval (0,1)
        Significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    value : float
        Currently only `value=0`, i.e. equality testing, is supported
    alternative : string, 'two-sided' (default), 'larger', 'smaller'
        Alternative hypothesis whether the power is calculated for a
        two-sided (default) or one sided test. In the case of a one-sided
        alternative, it is assumed that the test is in the appropriate tail.

    Returns
    -------
    nobs1 : float
        Number of observations in sample 1.
    """
    from statsmodels.stats.power import normal_sample_size_one_tail
    if alternative in ['two-sided', '2s']:
        alpha = alpha / 2
    _, std_null, std_alt = _std_2prop_power(diff, prop2, ratio=ratio, alpha=alpha, value=value)
    nobs = normal_sample_size_one_tail(diff, power, alpha, std_null=std_null, std_alternative=std_alt)
    return nobs