import numpy as np
from scipy import stats
from scipy.special import ncfdtrinc
from statsmodels.stats.power import ncf_cdf, ncf_ppf
from statsmodels.stats.robust_compare import TrimmedMean, scale_transform
from statsmodels.tools.testing import Holder
from statsmodels.stats.base import HolderTuple
def effectsize_oneway(means, vars_, nobs, use_var='unequal', ddof_between=0):
    """
    Effect size corresponding to Cohen's f = nc / nobs for oneway anova

    This contains adjustment for Welch and Brown-Forsythe Anova so that
    effect size can be used with FTestAnovaPower.

    Parameters
    ----------
    means : array_like
        Mean of samples to be compared
    vars_ : float or array_like
        Residual (within) variance of each sample or pooled
        If ``vars_`` is scalar, then it is interpreted as pooled variance that
        is the same for all samples, ``use_var`` will be ignored.
        Otherwise, the variances are used depending on the ``use_var`` keyword.
    nobs : int or array_like
        Number of observations for the samples.
        If nobs is scalar, then it is assumed that all samples have the same
        number ``nobs`` of observation, i.e. a balanced sample case.
        Otherwise, statistics will be weighted corresponding to nobs.
        Only relative sizes are relevant, any proportional change to nobs does
        not change the effect size.
    use_var : {"unequal", "equal", "bf"}
        If ``use_var`` is "unequal", then the variances can differ across
        samples and the effect size for Welch anova will be computed.
    ddof_between : int
        Degrees of freedom correction for the weighted between sum of squares.
        The denominator is ``nobs_total - ddof_between``
        This can be used to match differences across reference literature.

    Returns
    -------
    f2 : float
        Effect size corresponding to squared Cohen's f, which is also equal
        to the noncentrality divided by total number of observations.

    Notes
    -----
    This currently handles the following cases for oneway anova

    - balanced sample with homoscedastic variances
    - samples with different number of observations and with homoscedastic
      variances
    - samples with different number of observations and with heteroskedastic
      variances. This corresponds to Welch anova

    In the case of "unequal" and "bf" methods for unequal variances, the
    effect sizes do not directly correspond to the test statistic in Anova.
    Both have correction terms dropped or added, so the effect sizes match up
    with using FTestAnovaPower.
    If all variances are equal, then all three methods result in the same
    effect size. If variances are unequal, then the three methods produce
    small differences in effect size.

    Note, the effect size and power computation for BF Anova was not found in
    the literature. The correction terms were added so that FTestAnovaPower
    provides a good approximation to the power.

    Status: experimental
    We might add additional returns, if those are needed to support power
    and sample size applications.

    Examples
    --------
    The following shows how to compute effect size and power for each of the
    three anova methods. The null hypothesis is that the means are equal which
    corresponds to a zero effect size. Under the alternative, means differ
    with two sample means at a distance delta from the mean. We assume the
    variance is the same under the null and alternative hypothesis.

    ``nobs`` for the samples defines the fraction of observations in the
    samples. ``nobs`` in the power method defines the total sample size.

    In simulations, the computed power for standard anova,
    i.e.``use_var="equal"`` overestimates the simulated power by a few percent.
    The equal variance assumption does not hold in this example.

    >>> from statsmodels.stats.oneway import effectsize_oneway
    >>> from statsmodels.stats.power import FTestAnovaPower
    >>>
    >>> nobs = np.array([10, 12, 13, 15])
    >>> delta = 0.5
    >>> means_alt = np.array([-1, 0, 0, 1]) * delta
    >>> vars_ = np.arange(1, len(means_alt) + 1)
    >>>
    >>> f2_alt = effectsize_oneway(means_alt, vars_, nobs, use_var="equal")
    >>> f2_alt
    0.04581300813008131
    >>>
    >>> kwds = {'effect_size': np.sqrt(f2_alt), 'nobs': 100, 'alpha': 0.05,
    ...         'k_groups': 4}
    >>> power = FTestAnovaPower().power(**kwds)
    >>> power
    0.39165892158983273
    >>>
    >>> f2_alt = effectsize_oneway(means_alt, vars_, nobs, use_var="unequal")
    >>> f2_alt
    0.060640138408304504
    >>>
    >>> kwds['effect_size'] = np.sqrt(f2_alt)
    >>> power = FTestAnovaPower().power(**kwds)
    >>> power
    0.5047366512800622
    >>>
    >>> f2_alt = effectsize_oneway(means_alt, vars_, nobs, use_var="bf")
    >>> f2_alt
    0.04391324307956788
    >>>
    >>> kwds['effect_size'] = np.sqrt(f2_alt)
    >>> power = FTestAnovaPower().power(**kwds)
    >>> power
    0.3765792117047725

    """
    means = np.asarray(means)
    n_groups = means.shape[0]
    if np.size(nobs) == 1:
        nobs = np.ones(n_groups) * nobs
    nobs_t = nobs.sum()
    if use_var == 'equal':
        if np.size(vars_) == 1:
            var_resid = vars_
        else:
            vars_ = np.asarray(vars_)
            var_resid = ((nobs - 1) * vars_).sum() / (nobs_t - n_groups)
        vars_ = var_resid
    weights = nobs / vars_
    w_total = weights.sum()
    w_rel = weights / w_total
    meanw_t = w_rel @ means
    f2 = np.dot(weights, (means - meanw_t) ** 2) / (nobs_t - ddof_between)
    if use_var.lower() == 'bf':
        weights = nobs
        w_total = weights.sum()
        w_rel = weights / w_total
        meanw_t = w_rel @ means
        tmp = ((1.0 - nobs / nobs_t) * vars_).sum()
        statistic = 1.0 * (nobs * (means - meanw_t) ** 2).sum()
        statistic /= tmp
        f2 = statistic * (1.0 - nobs / nobs_t).sum() / nobs_t
        df_num2 = n_groups - 1
        df_num = tmp ** 2 / ((vars_ ** 2).sum() + (nobs / nobs_t * vars_).sum() ** 2 - 2 * (nobs / nobs_t * vars_ ** 2).sum())
        f2 *= df_num / df_num2
    return f2