import numpy as np
from scipy import stats
from scipy.special import ncfdtrinc
from statsmodels.stats.power import ncf_cdf, ncf_ppf
from statsmodels.stats.robust_compare import TrimmedMean, scale_transform
from statsmodels.tools.testing import Holder
from statsmodels.stats.base import HolderTuple
def power_equivalence_oneway(f2_alt, equiv_margin, nobs_t, n_groups=None, df=None, alpha=0.05, margin_type='f2'):
    """
    Power of  oneway equivalence test

    Parameters
    ----------
    f2_alt : float
        Effect size, squared Cohen's f, under the alternative.
    equiv_margin : float
        Equivalence margin in terms of effect size. Effect size can be chosen
        with `margin_type`. default is squared Cohen's f.
    nobs_t : ndarray
        Total number of observations summed over all groups.
    n_groups : int
        Number of groups in oneway comparison. If margin_type is "wellek",
        then either ``n_groups`` or ``df`` has to be given.
    df : tuple
        Degrees of freedom for F distribution,
        ``df = (n_groups - 1, nobs_t - n_groups)``
    alpha : float in (0, 1)
        Significance level for the hypothesis test.
    margin_type : "f2" or "wellek"
        Type of effect size used for equivalence margin, either squared
        Cohen's f or Wellek's psi. Default is "f2".

    Returns
    -------
    pow_alt : float
        Power of the equivalence test at given equivalence effect size under
        the alternative.
    """
    if df is None:
        if n_groups is None:
            raise ValueError('either df or n_groups has to be provided')
        df = (n_groups - 1, nobs_t - n_groups)
    if f2_alt == 0:
        f2_alt = 1e-13
    if margin_type in ['f2', 'fsqu', 'fsquared']:
        f2_null = equiv_margin
    elif margin_type == 'wellek':
        if n_groups is None:
            raise ValueError('If margin_type is wellek, then n_groups has to be provided')
        nobs_mean = nobs_t / n_groups
        f2_null = nobs_mean * equiv_margin ** 2 / nobs_t
        f2_alt = nobs_mean * f2_alt ** 2 / nobs_t
    else:
        raise ValueError('`margin_type` should be "f2" or "wellek"')
    crit_f_margin = ncf_ppf(alpha, df[0], df[1], nobs_t * f2_null)
    pwr_alt = ncf_cdf(crit_f_margin, df[0], df[1], nobs_t * f2_alt)
    return pwr_alt