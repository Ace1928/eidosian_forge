import numpy as np
from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import (
def rank_compare_2indep(x1, x2, use_t=True):
    """
    Statistics and tests for the probability that x1 has larger values than x2.

    p is the probability that a random draw from the population of
    the first sample has a larger value than a random draw from the
    population of the second sample, specifically

            p = P(x1 > x2) + 0.5 * P(x1 = x2)

    This is a measure underlying Wilcoxon-Mann-Whitney's U test,
    Fligner-Policello test and Brunner-Munzel test, and
    Inference is based on the asymptotic distribution of the Brunner-Munzel
    test. The half probability for ties corresponds to the use of midranks
    and make it valid for discrete variables.

    The Null hypothesis for stochastic equality is p = 0.5, which corresponds
    to the Brunner-Munzel test.

    Parameters
    ----------
    x1, x2 : array_like
        Array of samples, should be one-dimensional.
    use_t : boolean
        If use_t is true, the t distribution with Welch-Satterthwaite type
        degrees of freedom is used for p-value and confidence interval.
        If use_t is false, then the normal distribution is used.

    Returns
    -------
    res : RankCompareResult
        The results instance contains the results for the Brunner-Munzel test
        and has methods for hypothesis tests, confidence intervals and summary.

        statistic : float
            The Brunner-Munzel W statistic.
        pvalue : float
            p-value assuming an t distribution. One-sided or
            two-sided, depending on the choice of `alternative` and `use_t`.

    See Also
    --------
    RankCompareResult
    scipy.stats.brunnermunzel : Brunner-Munzel test for stochastic equality
    scipy.stats.mannwhitneyu : Mann-Whitney rank test on two samples.

    Notes
    -----
    Wilcoxon-Mann-Whitney assumes equal variance or equal distribution under
    the Null hypothesis. Fligner-Policello test allows for unequal variances
    but assumes continuous distribution, i.e. no ties.
    Brunner-Munzel extend the test to allow for unequal variance and discrete
    or ordered categorical random variables.

    Brunner and Munzel recommended to estimate the p-value by t-distribution
    when the size of data is 50 or less. If the size is lower than 10, it would
    be better to use permuted Brunner Munzel test (see [2]_) for the test
    of stochastic equality.

    This measure has been introduced in the literature under many different
    names relying on a variety of assumptions.
    In psychology, McGraw and Wong (1992) introduced it as Common Language
    effect size for the continuous, normal distribution case,
    Vargha and Delaney (2000) [3]_ extended it to the nonparametric
    continuous distribution case as in Fligner-Policello.

    WMW and related tests can only be interpreted as test of medians or tests
    of central location only under very restrictive additional assumptions
    such as both distribution are identical under the equality null hypothesis
    (assumed by Mann-Whitney) or both distributions are symmetric (shown by
    Fligner-Policello). If the distribution of the two samples can differ in
    an arbitrary way, then the equality Null hypothesis corresponds to p=0.5
    against an alternative p != 0.5.  see for example Conroy (2012) [4]_ and
    Divine et al (2018) [5]_ .

    Note: Brunner-Munzel and related literature define the probability that x1
    is stochastically smaller than x2, while here we use stochastically larger.
    This equivalent to switching x1 and x2 in the two sample case.

    References
    ----------
    .. [1] Brunner, E. and Munzel, U. "The nonparametric Benhrens-Fisher
           problem: Asymptotic theory and a small-sample approximation".
           Biometrical Journal. Vol. 42(2000): 17-25.
    .. [2] Neubert, K. and Brunner, E. "A studentized permutation test for the
           non-parametric Behrens-Fisher problem". Computational Statistics and
           Data Analysis. Vol. 51(2007): 5192-5204.
    .. [3] Vargha, András, and Harold D. Delaney. 2000. “A Critique and
           Improvement of the CL Common Language Effect Size Statistics of
           McGraw and Wong.” Journal of Educational and Behavioral Statistics
           25 (2): 101–32. https://doi.org/10.3102/10769986025002101.
    .. [4] Conroy, Ronán M. 2012. “What Hypotheses Do ‘Nonparametric’ Two-Group
           Tests Actually Test?” The Stata Journal: Promoting Communications on
           Statistics and Stata 12 (2): 182–90.
           https://doi.org/10.1177/1536867X1201200202.
    .. [5] Divine, George W., H. James Norton, Anna E. Barón, and Elizabeth
           Juarez-Colunga. 2018. “The Wilcoxon–Mann–Whitney Procedure Fails as
           a Test of Medians.” The American Statistician 72 (3): 278–86.
           https://doi.org/10.1080/00031305.2017.1305291.

    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    nobs1 = len(x1)
    nobs2 = len(x2)
    nobs = nobs1 + nobs2
    if nobs1 == 0 or nobs2 == 0:
        raise ValueError('one sample has zero length')
    rank1, rank2, ranki1, ranki2 = rankdata_2samp(x1, x2)
    meanr1 = np.mean(rank1, axis=0)
    meanr2 = np.mean(rank2, axis=0)
    meanri1 = np.mean(ranki1, axis=0)
    meanri2 = np.mean(ranki2, axis=0)
    S1 = np.sum(np.power(rank1 - ranki1 - meanr1 + meanri1, 2.0), axis=0)
    S1 /= nobs1 - 1
    S2 = np.sum(np.power(rank2 - ranki2 - meanr2 + meanri2, 2.0), axis=0)
    S2 /= nobs2 - 1
    wbfn = nobs1 * nobs2 * (meanr1 - meanr2)
    wbfn /= (nobs1 + nobs2) * np.sqrt(nobs1 * S1 + nobs2 * S2)
    if use_t:
        df_numer = np.power(nobs1 * S1 + nobs2 * S2, 2.0)
        df_denom = np.power(nobs1 * S1, 2.0) / (nobs1 - 1)
        df_denom += np.power(nobs2 * S2, 2.0) / (nobs2 - 1)
        df = df_numer / df_denom
        pvalue = 2 * stats.t.sf(np.abs(wbfn), df)
    else:
        pvalue = 2 * stats.norm.sf(np.abs(wbfn))
        df = None
    var1 = S1 / (nobs - nobs1) ** 2
    var2 = S2 / (nobs - nobs2) ** 2
    var_prob = var1 / nobs1 + var2 / nobs2
    var = nobs * (var1 / nobs1 + var2 / nobs2)
    prob1 = (meanr1 - (nobs1 + 1) / 2) / nobs2
    prob2 = (meanr2 - (nobs2 + 1) / 2) / nobs1
    return RankCompareResult(statistic=wbfn, pvalue=pvalue, s1=S1, s2=S2, var1=var1, var2=var2, var=var, var_prob=var_prob, nobs1=nobs1, nobs2=nobs2, nobs=nobs, mean1=meanr1, mean2=meanr2, prob1=prob1, prob2=prob2, somersd1=prob1 * 2 - 1, somersd2=prob2 * 2 - 1, df=df, use_t=use_t)