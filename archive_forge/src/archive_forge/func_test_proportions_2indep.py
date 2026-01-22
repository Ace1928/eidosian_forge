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
def test_proportions_2indep(count1, nobs1, count2, nobs2, value=None, method=None, compare='diff', alternative='two-sided', correction=True, return_results=True):
    """
    Hypothesis test for comparing two independent proportions

    This assumes that we have two independent binomial samples.

    The Null and alternative hypothesis are

    for compare = 'diff'

    - H0: prop1 - prop2 - value = 0
    - H1: prop1 - prop2 - value != 0  if alternative = 'two-sided'
    - H1: prop1 - prop2 - value > 0   if alternative = 'larger'
    - H1: prop1 - prop2 - value < 0   if alternative = 'smaller'

    for compare = 'ratio'

    - H0: prop1 / prop2 - value = 0
    - H1: prop1 / prop2 - value != 0  if alternative = 'two-sided'
    - H1: prop1 / prop2 - value > 0   if alternative = 'larger'
    - H1: prop1 / prop2 - value < 0   if alternative = 'smaller'

    for compare = 'odds-ratio'

    - H0: or - value = 0
    - H1: or - value != 0  if alternative = 'two-sided'
    - H1: or - value > 0   if alternative = 'larger'
    - H1: or - value < 0   if alternative = 'smaller'

    where odds-ratio or = prop1 / (1 - prop1) / (prop2 / (1 - prop2))

    Parameters
    ----------
    count1 : int
        Count for first sample.
    nobs1 : int
        Sample size for first sample.
    count2 : int
        Count for the second sample.
    nobs2 : int
        Sample size for the second sample.
    value : float
        Value of the difference, risk ratio or odds ratio of 2 independent
        proportions under the null hypothesis.
        Default is equal proportions, 0 for diff and 1 for risk-ratio and for
        odds-ratio.
    method : string
        Method for computing the hypothesis test. If method is None, then a
        default method is used. The default might change as more methods are
        added.

        diff:

        - 'wald',
        - 'agresti-caffo'
        - 'score' if correction is True, then this uses the degrees of freedom
           correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985

        ratio:

        - 'log': wald test using log transformation
        - 'log-adjusted': wald test using log transformation,
           adds 0.5 to counts
        - 'score': if correction is True, then this uses the degrees of freedom
           correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985

        odds-ratio:

        - 'logit': wald test using logit transformation
        - 'logit-adjusted': wald test using logit transformation,
           adds 0.5 to counts
        - 'logit-smoothed': wald test using logit transformation, biases
           cell counts towards independence by adding two observations in
           total.
        - 'score' if correction is True, then this uses the degrees of freedom
           correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985

    compare : {'diff', 'ratio' 'odds-ratio'}
        If compare is `diff`, then the hypothesis test is for the risk
        difference diff = p1 - p2.
        If compare is `ratio`, then the hypothesis test is for the
        risk ratio defined by ratio = p1 / p2.
        If compare is `odds-ratio`, then the hypothesis test is for the
        odds-ratio defined by or = p1 / (1 - p1) / (p2 / (1 - p2)
    alternative : {'two-sided', 'smaller', 'larger'}
        alternative hypothesis, which can be two-sided or either one of the
        one-sided tests.
    correction : bool
        If correction is True (default), then the Miettinen and Nurminen
        small sample correction to the variance nobs / (nobs - 1) is used.
        Applies only if method='score'.
    return_results : bool
        If true, then a results instance with extra information is returned,
        otherwise a tuple with statistic and pvalue is returned.

    Returns
    -------
    results : results instance or tuple
        If return_results is True, then a results instance with the
        information in attributes is returned.
        If return_results is False, then only ``statistic`` and ``pvalue``
        are returned.

        statistic : float
            test statistic asymptotically normal distributed N(0, 1)
        pvalue : float
            p-value based on normal distribution
        other attributes :
            additional information about the hypothesis test

    See Also
    --------
    tost_proportions_2indep
    confint_proportions_2indep

    Notes
    -----
    Status: experimental, API and defaults might still change.
        More ``methods`` will be added.

    The current default methods are

    - 'diff': 'agresti-caffo',
    - 'ratio': 'log-adjusted',
    - 'odds-ratio': 'logit-adjusted'

    """
    method_default = {'diff': 'agresti-caffo', 'ratio': 'log-adjusted', 'odds-ratio': 'logit-adjusted'}
    if compare.lower() == 'or':
        compare = 'odds-ratio'
    if method is None:
        method = method_default[compare]
    method = method.lower()
    if method.startswith('agr'):
        method = 'agresti-caffo'
    if value is None:
        value = 0 if compare == 'diff' else 1
    count1, nobs1, count2, nobs2 = map(np.asarray, [count1, nobs1, count2, nobs2])
    p1 = count1 / nobs1
    p2 = count2 / nobs2
    diff = p1 - p2
    ratio = p1 / p2
    odds_ratio = p1 / (1 - p1) / p2 * (1 - p2)
    res = None
    if compare == 'diff':
        if method in ['wald', 'agresti-caffo']:
            addone = 1 if method == 'agresti-caffo' else 0
            count1_, nobs1_ = (count1 + addone, nobs1 + 2 * addone)
            count2_, nobs2_ = (count2 + addone, nobs2 + 2 * addone)
            p1_ = count1_ / nobs1_
            p2_ = count2_ / nobs2_
            diff_stat = p1_ - p2_ - value
            var = p1_ * (1 - p1_) / nobs1_ + p2_ * (1 - p2_) / nobs2_
            statistic = diff_stat / np.sqrt(var)
            distr = 'normal'
        elif method.startswith('newcomb'):
            msg = 'newcomb not available for hypothesis test'
            raise NotImplementedError(msg)
        elif method == 'score':
            res = score_test_proportions_2indep(count1, nobs1, count2, nobs2, value=value, compare=compare, alternative=alternative, correction=correction, return_results=return_results)
            if return_results is False:
                statistic, pvalue = res[:2]
            distr = 'normal'
            diff_stat = None
        else:
            raise ValueError('method not recognized')
    elif compare == 'ratio':
        if method in ['log', 'log-adjusted']:
            addhalf = 0.5 if method == 'log-adjusted' else 0
            count1_, nobs1_ = (count1 + addhalf, nobs1 + addhalf)
            count2_, nobs2_ = (count2 + addhalf, nobs2 + addhalf)
            p1_ = count1_ / nobs1_
            p2_ = count2_ / nobs2_
            ratio_ = p1_ / p2_
            var = 1 / count1_ - 1 / nobs1_ + 1 / count2_ - 1 / nobs2_
            diff_stat = np.log(ratio_) - np.log(value)
            statistic = diff_stat / np.sqrt(var)
            distr = 'normal'
        elif method == 'score':
            res = score_test_proportions_2indep(count1, nobs1, count2, nobs2, value=value, compare=compare, alternative=alternative, correction=correction, return_results=return_results)
            if return_results is False:
                statistic, pvalue = res[:2]
            distr = 'normal'
            diff_stat = None
        else:
            raise ValueError('method not recognized')
    elif compare == 'odds-ratio':
        if method in ['logit', 'logit-adjusted', 'logit-smoothed']:
            if method in ['logit-smoothed']:
                adjusted = _shrink_prob(count1, nobs1, count2, nobs2, shrink_factor=2, return_corr=False)[0]
                count1_, nobs1_, count2_, nobs2_ = adjusted
            else:
                addhalf = 0.5 if method == 'logit-adjusted' else 0
                count1_, nobs1_ = (count1 + addhalf, nobs1 + 2 * addhalf)
                count2_, nobs2_ = (count2 + addhalf, nobs2 + 2 * addhalf)
            p1_ = count1_ / nobs1_
            p2_ = count2_ / nobs2_
            odds_ratio_ = p1_ / (1 - p1_) / p2_ * (1 - p2_)
            var = 1 / count1_ + 1 / (nobs1_ - count1_) + 1 / count2_ + 1 / (nobs2_ - count2_)
            diff_stat = np.log(odds_ratio_) - np.log(value)
            statistic = diff_stat / np.sqrt(var)
            distr = 'normal'
        elif method == 'score':
            res = score_test_proportions_2indep(count1, nobs1, count2, nobs2, value=value, compare=compare, alternative=alternative, correction=correction, return_results=return_results)
            if return_results is False:
                statistic, pvalue = res[:2]
            distr = 'normal'
            diff_stat = None
        else:
            raise ValueError('method "%s" not recognized' % method)
    else:
        raise ValueError('compare "%s" not recognized' % compare)
    if distr == 'normal' and diff_stat is not None:
        statistic, pvalue = _zstat_generic2(diff_stat, np.sqrt(var), alternative=alternative)
    if return_results:
        if res is None:
            res = HolderTuple(statistic=statistic, pvalue=pvalue, compare=compare, method=method, diff=diff, ratio=ratio, odds_ratio=odds_ratio, variance=var, alternative=alternative, value=value)
        else:
            res.diff = diff
            res.ratio = ratio
            res.odds_ratio = odds_ratio
            res.value = value
        return res
    else:
        return (statistic, pvalue)