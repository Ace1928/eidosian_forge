import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.base import HolderTuple
def conf_int_samples(self, alpha=0.05, use_t=None, nobs=None, ci_func=None):
    """confidence intervals for the effect size estimate of samples

        Additional information needs to be provided for confidence intervals
        that are not based on normal distribution using available variance.
        This is likely to change in future.

        Parameters
        ----------
        alpha : float in (0, 1)
            Significance level for confidence interval. Nominal coverage is
            ``1 - alpha``.
        use_t : None or bool
            If use_t is None, then the attribute `use_t` determines whether
            normal or t-distribution is used for confidence intervals.
            Specifying use_t overrides the attribute.
            If use_t is false, then confidence intervals are based on the
            normal distribution. If it is true, then the t-distribution is
            used.
        nobs : None or float
            Number of observations used for degrees of freedom computation.
            Only used if use_t is true.
        ci_func : None or callable
            User provided function to compute confidence intervals.
            This is not used yet and will allow using non-standard confidence
            intervals.

        Returns
        -------
        ci_eff : tuple of ndarrays
            Tuple (ci_low, ci_upp) with confidence interval computed for each
            sample.

        Notes
        -----
        CombineResults currently only has information from the combine_effects
        function, which does not provide details about individual samples.
        """
    if (alpha, use_t) in self.cache_ci:
        return self.cache_ci[alpha, use_t]
    if use_t is None:
        use_t = self.use_t
    if ci_func is not None:
        kwds = {'use_t': use_t} if use_t is not None else {}
        ci_eff = ci_func(alpha=alpha, **kwds)
        self.ci_sample_distr = 'ci_func'
    else:
        if use_t is False:
            crit = stats.norm.isf(alpha / 2)
            self.ci_sample_distr = 'normal'
        elif nobs is not None:
            df_resid = nobs - 1
            crit = stats.t.isf(alpha / 2, df_resid)
            self.ci_sample_distr = 't'
        else:
            msg = '`use_t=True` requires `nobs` for each sample or `ci_func`. Using normal distribution for confidence interval of individual samples.'
            import warnings
            warnings.warn(msg)
            crit = stats.norm.isf(alpha / 2)
            self.ci_sample_distr = 'normal'
        ci_low = self.eff - crit * self.sd_eff
        ci_upp = self.eff + crit * self.sd_eff
        ci_eff = (ci_low, ci_upp)
    self.cache_ci[alpha, use_t] = ci_eff
    return ci_eff