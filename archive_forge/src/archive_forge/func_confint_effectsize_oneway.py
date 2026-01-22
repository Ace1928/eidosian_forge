import numpy as np
from scipy import stats
from scipy.special import ncfdtrinc
from statsmodels.stats.power import ncf_cdf, ncf_ppf
from statsmodels.stats.robust_compare import TrimmedMean, scale_transform
from statsmodels.tools.testing import Holder
from statsmodels.stats.base import HolderTuple
def confint_effectsize_oneway(f_stat, df, alpha=0.05, nobs=None):
    """
    Confidence interval for effect size in oneway anova for F distribution

    This does not yet handle non-negativity constraint on nc.
    Currently only two-sided alternative is supported.

    Parameters
    ----------
    f_stat : float
    df : tuple
        degrees of freedom ``df = (df1, df2)`` where

        - df1 : numerator degrees of freedom, number of constraints
        - df2 : denominator degrees of freedom, df_resid

    alpha : float, default 0.05
    nobs : int, default None

    Returns
    -------
    Holder
        Class with effect size and confidence attributes

    Notes
    -----
    The confidence interval for the noncentrality parameter is obtained by
    inverting the cdf of the noncentral F distribution. Confidence intervals
    for other effect sizes are computed by endpoint transformation.


    R package ``effectsize`` does not compute the confidence intervals in the
    same way. Their confidence intervals can be replicated with

    >>> ci_nc = confint_noncentrality(f_stat, df1, df2, alpha=0.1)
    >>> ci_es = smo._fstat2effectsize(ci_nc / df1, df1, df2)

    See Also
    --------
    confint_noncentrality
    """
    df1, df2 = df
    if nobs is None:
        nobs = df1 + df2 + 1
    ci_nc = confint_noncentrality(f_stat, df, alpha=alpha)
    ci_f2 = ci_nc / nobs
    ci_res = convert_effectsize_fsqu(f2=ci_f2)
    ci_res.ci_omega2 = (ci_f2 - df1 / df2) / (ci_f2 + 1 + 1 / df2)
    ci_res.ci_nc = ci_nc
    ci_res.ci_f = np.sqrt(ci_res.f2)
    ci_res.ci_eta = np.sqrt(ci_res.eta2)
    ci_res.ci_f_corrected = np.sqrt(ci_res.f2 * (df1 + 1) / df1)
    return ci_res