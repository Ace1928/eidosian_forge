import warnings
import numpy as np
from scipy import stats, optimize, special
from statsmodels.tools.rootfinding import brentq_expanding
def ttest_power(effect_size, nobs, alpha, df=None, alternative='two-sided'):
    """Calculate power of a ttest
    """
    d = effect_size
    if df is None:
        df = nobs - 1
    if alternative in ['two-sided', '2s']:
        alpha_ = alpha / 2.0
    elif alternative in ['smaller', 'larger']:
        alpha_ = alpha
    else:
        raise ValueError("alternative has to be 'two-sided', 'larger' " + "or 'smaller'")
    pow_ = 0
    if alternative in ['two-sided', '2s', 'larger']:
        crit_upp = stats.t.isf(alpha_, df)
        if np.any(np.isnan(crit_upp)):
            pow_ = np.nan
        else:
            pow_ = nct_sf(crit_upp, df, d * np.sqrt(nobs))
    if alternative in ['two-sided', '2s', 'smaller']:
        crit_low = stats.t.ppf(alpha_, df)
        if np.any(np.isnan(crit_low)):
            pow_ = np.nan
        else:
            pow_ += nct_cdf(crit_low, df, d * np.sqrt(nobs))
    return pow_