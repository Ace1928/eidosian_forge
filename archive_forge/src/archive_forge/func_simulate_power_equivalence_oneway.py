import numpy as np
from scipy import stats
from scipy.special import ncfdtrinc
from statsmodels.stats.power import ncf_cdf, ncf_ppf
from statsmodels.stats.robust_compare import TrimmedMean, scale_transform
from statsmodels.tools.testing import Holder
from statsmodels.stats.base import HolderTuple
def simulate_power_equivalence_oneway(means, nobs, equiv_margin, vars_=None, k_mc=1000, trim_frac=0, options_var=None, margin_type='f2'):
    """Simulate Power for oneway equivalence test (Wellek's Anova)

    This function is experimental and written to evaluate asymptotic power
    function. This function will change without backwards compatibility
    constraints. The only part that is stable is `pvalue` attribute in results.

    Effect size for equivalence margin

    """
    if options_var is None:
        options_var = ['unequal', 'equal', 'bf']
    if vars_ is not None:
        stds = np.sqrt(vars_)
    else:
        stds = np.ones(len(means))
    nobs_mean = nobs.mean()
    n_groups = len(nobs)
    res_mc = []
    f_mc = []
    reject_mc = []
    other_mc = []
    for _ in range(k_mc):
        y0, y1, y2, y3 = (m + std * np.random.randn(n) for n, m, std in zip(nobs, means, stds))
        res_i = []
        f_i = []
        reject_i = []
        other_i = []
        for uv in options_var:
            res0 = anova_oneway([y0, y1, y2, y3], use_var=uv, trim_frac=trim_frac)
            f_stat = res0.statistic
            res1 = equivalence_oneway_generic(f_stat, n_groups, nobs.sum(), equiv_margin, res0.df, alpha=0.05, margin_type=margin_type)
            res_i.append(res1.pvalue)
            es_wellek = f_stat * (n_groups - 1) / nobs_mean
            f_i.append(es_wellek)
            reject_i.append(res1.reject)
            other_i.extend([res1.crit_f, res1.crit_es, res1.power_zero])
        res_mc.append(res_i)
        f_mc.append(f_i)
        reject_mc.append(reject_i)
        other_mc.append(other_i)
    f_mc = np.asarray(f_mc)
    other_mc = np.asarray(other_mc)
    res_mc = np.asarray(res_mc)
    reject_mc = np.asarray(reject_mc)
    res = Holder(f_stat=f_mc, other=other_mc, pvalue=res_mc, reject=reject_mc)
    return res