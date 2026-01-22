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
def waic(data, pointwise=None, var_name=None, scale=None, dask_kwargs=None):
    """Compute the widely applicable information criterion.

    Estimates the expected log pointwise predictive density (elpd) using WAIC. Also calculates the
    WAIC's standard error and the effective number of parameters.
    Read more theory here https://arxiv.org/abs/1507.04544 and here https://arxiv.org/abs/1004.2316

    Parameters
    ----------
    data: obj
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Refer to documentation of :func:`arviz.convert_to_inference_data` for details.
    pointwise: bool
        If True the pointwise predictive accuracy will be returned. Defaults to
        ``stats.ic_pointwise`` rcParam.
    var_name : str, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for waic computation.
    scale: str
        Output scale for WAIC. Available options are:

        - `log` : (default) log-score
        - `negative_log` : -1 * log-score
        - `deviance` : -2 * log-score

        A higher log-score (or a lower deviance or negative log_score) indicates a model with
        better predictive accuracy.
    dask_kwargs : dict, optional
        Dask related kwargs passed to :func:`~arviz.wrap_xarray_ufunc`.

    Returns
    -------
    ELPDData object (inherits from :class:`pandas.Series`) with the following row/attributes:
    elpd_waic: approximated expected log pointwise predictive density (elpd)
    se: standard error of the elpd
    p_waic: effective number parameters
    var_warn: bool
        True if posterior variance of the log predictive densities exceeds 0.4
    waic_i: :class:`~xarray.DataArray` with the pointwise predictive accuracy,
            only if pointwise=True
    scale: scale of the elpd

        The returned object has a custom print method that overrides pd.Series method.

    See Also
    --------
    loo : Compute Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO-CV).
    compare : Compare models based on PSIS-LOO-CV or WAIC.
    plot_compare : Summary plot for model comparison.

    Examples
    --------
    Calculate WAIC of a model:

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data("centered_eight")
           ...: az.waic(data)

    Calculate WAIC of a model and return the pointwise values:

    .. ipython::

        In [2]: data_waic = az.waic(data, pointwise=True)
           ...: data_waic.waic_i
    """
    inference_data = convert_to_inference_data(data)
    log_likelihood = _get_log_likelihood(inference_data, var_name=var_name)
    scale = rcParams['stats.ic_scale'] if scale is None else scale.lower()
    pointwise = rcParams['stats.ic_pointwise'] if pointwise is None else pointwise
    if scale == 'deviance':
        scale_value = -2
    elif scale == 'log':
        scale_value = 1
    elif scale == 'negative_log':
        scale_value = -1
    else:
        raise TypeError('Valid scale values are "deviance", "log", "negative_log"')
    log_likelihood = log_likelihood.stack(__sample__=('chain', 'draw'))
    shape = log_likelihood.shape
    n_samples = shape[-1]
    n_data_points = np.prod(shape[:-1])
    ufunc_kwargs = {'n_dims': 1, 'ravel': False}
    kwargs = {'input_core_dims': [['__sample__']]}
    lppd_i = _wrap_xarray_ufunc(_logsumexp, log_likelihood, func_kwargs={'b_inv': n_samples}, ufunc_kwargs=ufunc_kwargs, dask_kwargs=dask_kwargs, **kwargs)
    vars_lpd = log_likelihood.var(dim='__sample__')
    warn_mg = False
    if np.any(vars_lpd > 0.4):
        warnings.warn('For one or more samples the posterior variance of the log predictive densities exceeds 0.4. This could be indication of WAIC starting to fail. \nSee http://arxiv.org/abs/1507.04544 for details')
        warn_mg = True
    waic_i = scale_value * (lppd_i - vars_lpd)
    waic_se = (n_data_points * np.var(waic_i.values)) ** 0.5
    waic_sum = np.sum(waic_i.values)
    p_waic = np.sum(vars_lpd.values)
    if not pointwise:
        return ELPDData(data=[waic_sum, waic_se, p_waic, n_samples, n_data_points, warn_mg, scale], index=['waic', 'se', 'p_waic', 'n_samples', 'n_data_points', 'warning', 'scale'])
    if np.equal(waic_sum, waic_i).all():
        warnings.warn('The point-wise WAIC is the same with the sum WAIC, please double check\n            the Observed RV in your model to make sure it returns element-wise logp.\n            ')
    return ELPDData(data=[waic_sum, waic_se, p_waic, n_samples, n_data_points, warn_mg, waic_i.rename('waic_i'), scale], index=['elpd_waic', 'se', 'p_waic', 'n_samples', 'n_data_points', 'warning', 'waic_i', 'scale'])