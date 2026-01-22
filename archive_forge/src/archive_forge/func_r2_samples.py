import itertools
import warnings
from copy import deepcopy
from typing import List, Optional, Tuple, Union, Mapping, cast, Callable
import numpy as np
import pandas as pd
import scipy.stats as st
from xarray_einstats import stats
import xarray as xr
from scipy.optimize import minimize
from typing_extensions import Literal
from .. import _log
from ..data import InferenceData, convert_to_dataset, convert_to_inference_data, extract
from ..rcparams import rcParams, ScaleKeyword, ICKeyword
from ..utils import Numba, _numba_var, _var_names, get_coords
from .density_utils import get_bins as _get_bins
from .density_utils import histogram as _histogram
from .density_utils import kde as _kde
from .diagnostics import _mc_error, _multichain_statistics, ess
from .stats_utils import ELPDData, _circular_standard_deviation, smooth_data
from .stats_utils import get_log_likelihood as _get_log_likelihood
from .stats_utils import get_log_prior as _get_log_prior
from .stats_utils import logsumexp as _logsumexp
from .stats_utils import make_ufunc as _make_ufunc
from .stats_utils import stats_variance_2d as svar
from .stats_utils import wrap_xarray_ufunc as _wrap_xarray_ufunc
from ..sel_utils import xarray_var_iter
from ..labels import BaseLabeller
def r2_samples(y_true, y_pred):
    """R² samples for Bayesian regression models. Only valid for linear models.

    Parameters
    ----------
    y_true: array-like of shape = (n_outputs,)
        Ground truth (correct) target values.
    y_pred: array-like of shape = (n_posterior_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    Pandas Series with the following indices:
    Bayesian R² samples.

    See Also
    --------
    plot_lm : Posterior predictive and mean plots for regression-like data.

    Examples
    --------
    Calculate R² samples for Bayesian regression models :

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data('regression1d')
           ...: y_true = data.observed_data["y"].values
           ...: y_pred = data.posterior_predictive.stack(sample=("chain", "draw"))["y"].values.T
           ...: az.r2_samples(y_true, y_pred)

    """
    _numba_flag = Numba.numba_flag
    if y_pred.ndim == 1:
        var_y_est = _numba_var(svar, np.var, y_pred)
        var_e = _numba_var(svar, np.var, y_true - y_pred)
    else:
        var_y_est = _numba_var(svar, np.var, y_pred, axis=1)
        var_e = _numba_var(svar, np.var, y_true - y_pred, axis=1)
    r_squared = var_y_est / (var_y_est + var_e)
    return r_squared