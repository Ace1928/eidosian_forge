import numpy as np
from scipy import stats
from scipy.special import ncfdtrinc
from statsmodels.stats.power import ncf_cdf, ncf_ppf
from statsmodels.stats.robust_compare import TrimmedMean, scale_transform
from statsmodels.tools.testing import Holder
from statsmodels.stats.base import HolderTuple
def equivalence_oneway(data, equiv_margin, groups=None, use_var='unequal', welch_correction=True, trim_frac=0, margin_type='f2'):
    """equivalence test for oneway anova (Wellek's Anova)

    The null hypothesis is that the means differ by more than `equiv_margin`
    in the anova distance measure.
    If the Null is rejected, then the data supports that means are equivalent,
    i.e. within a given distance.

    Parameters
    ----------
    data : tuple of array_like or DataFrame or Series
        Data for k independent samples, with k >= 2.
        The data can be provided as a tuple or list of arrays or in long
        format with outcome observations in ``data`` and group membership in
        ``groups``.
    equiv_margin : float
        Equivalence margin in terms of effect size. Effect size can be chosen
        with `margin_type`. default is squared Cohen's f.
    groups : ndarray or Series
        If data is in long format, then groups is needed as indicator to which
        group or sample and observations belongs.
    use_var : {"unequal", "equal" or "bf"}
        `use_var` specified how to treat heteroscedasticity, unequal variance,
        across samples. Three approaches are available

        "unequal" : Variances are not assumed to be equal across samples.
            Heteroscedasticity is taken into account with Welch Anova and
            Satterthwaite-Welch degrees of freedom.
            This is the default.
        "equal" : Variances are assumed to be equal across samples.
            This is the standard Anova.
        "bf: Variances are not assumed to be equal across samples.
            The method is Browne-Forsythe (1971) for testing equality of means
            with the corrected degrees of freedom by Merothra. The original BF
            degrees of freedom are available as additional attributes in the
            results instance, ``df_denom2`` and ``p_value2``.

    welch_correction : bool
        If this is false, then the Welch correction to the test statistic is
        not included. This allows the computation of an effect size measure
        that corresponds more closely to Cohen's f.
    trim_frac : float in [0, 0.5)
        Optional trimming for Anova with trimmed mean and winsorized variances.
        With the default trim_frac equal to zero, the oneway Anova statistics
        are computed without trimming. If `trim_frac` is larger than zero,
        then the largest and smallest observations in each sample are trimmed.
        The number of trimmed observations is the fraction of number of
        observations in the sample truncated to the next lower integer.
        `trim_frac` has to be smaller than 0.5, however, if the fraction is
        so large that there are not enough observations left over, then `nan`
        will be returned.
    margin_type : "f2" or "wellek"
        Type of effect size used for equivalence margin, either squared
        Cohen's f or Wellek's psi. Default is "f2".

    Returns
    -------
    results : instance of HolderTuple class
        The two main attributes are test statistic `statistic` and p-value
        `pvalue`.

    See Also
    --------
    anova_oneway
    equivalence_scale_oneway
    """
    res0 = anova_oneway(data, groups=groups, use_var=use_var, welch_correction=welch_correction, trim_frac=trim_frac)
    f_stat = res0.statistic
    res = equivalence_oneway_generic(f_stat, res0.n_groups, res0.nobs_t, equiv_margin, res0.df, alpha=0.05, margin_type=margin_type)
    return res