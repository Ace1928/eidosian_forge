from functools import partial
import numpy as np
from scipy import stats
from statsmodels.tools.validation import string_like
from ._lilliefors_critical_values import (critical_values,
from .tabledist import TableDist
def get_lilliefors_table(dist='norm'):
    """
    Generates tables for significance levels of Lilliefors test statistics

    Tables for available normal and exponential distribution testing,
    as specified in Lilliefors references above

    Parameters
    ----------
    dist : str
        distribution being tested in set {'norm', 'exp'}.

    Returns
    -------
    lf : TableDist object.
        table of critical values
    """
    alpha = 1 - np.array(PERCENTILES) / 100.0
    alpha = alpha[::-1]
    dist = 'normal' if dist == 'norm' else dist
    if dist not in critical_values:
        raise ValueError("Invalid dist parameter. Must be 'norm' or 'exp'")
    cv_data = critical_values[dist]
    acv_data = asymp_critical_values[dist]
    size = np.array(sorted(cv_data), dtype=float)
    crit_lf = np.array([cv_data[key] for key in sorted(cv_data)])
    crit_lf = crit_lf[:, ::-1]
    asym_params = np.array([acv_data[key] for key in sorted(acv_data)])
    asymp_fn = _make_asymptotic_function(asym_params[::-1])
    lf = TableDist(alpha, size, crit_lf, asymptotic=asymp_fn)
    return lf