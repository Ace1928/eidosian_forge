import warnings
from collections.abc import Sequence
import numpy as np
import packaging
import pandas as pd
import scipy
from scipy import stats
from ..data import convert_to_dataset
from ..utils import Numba, _numba_var, _stack, _var_names
from .density_utils import histogram as _histogram
from .stats_utils import _circular_standard_deviation, _sqrt
from .stats_utils import autocov as _autocov
from .stats_utils import not_valid as _not_valid
from .stats_utils import quantile as _quantile
from .stats_utils import stats_variance_2d as svar
from .stats_utils import wrap_xarray_ufunc as _wrap_xarray_ufunc
def ks_summary(pareto_tail_indices):
    """Display a summary of Pareto tail indices.

    Parameters
    ----------
    pareto_tail_indices : array
      Pareto tail indices.

    Returns
    -------
    df_k : dataframe
      Dataframe containing k diagnostic values.
    """
    _numba_flag = Numba.numba_flag
    if _numba_flag:
        bins = np.asarray([-np.inf, 0.5, 0.7, 1, np.inf])
        kcounts, *_ = _histogram(pareto_tail_indices, bins)
    else:
        kcounts, *_ = _histogram(pareto_tail_indices, bins=[-np.inf, 0.5, 0.7, 1, np.inf])
    kprop = kcounts / len(pareto_tail_indices) * 100
    df_k = pd.DataFrame(dict(_=['(good)', '(ok)', '(bad)', '(very bad)'], Count=kcounts, Pct=kprop)).rename(index={0: '(-Inf, 0.5]', 1: ' (0.5, 0.7]', 2: '   (0.7, 1]', 3: '   (1, Inf)'})
    if np.sum(kcounts[1:]) == 0:
        warnings.warn('All Pareto k estimates are good (k < 0.5)')
    elif np.sum(kcounts[2:]) == 0:
        warnings.warn('All Pareto k estimates are ok (k < 0.7)')
    return df_k