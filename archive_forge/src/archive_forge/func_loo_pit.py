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
def loo_pit(idata=None, *, y=None, y_hat=None, log_weights=None):
    """Compute leave one out (PSIS-LOO) probability integral transform (PIT) values.

    Parameters
    ----------
    idata: InferenceData
        :class:`arviz.InferenceData` object.
    y: array, DataArray or str
        Observed data. If str, ``idata`` must be present and contain the observed data group
    y_hat: array, DataArray or str
        Posterior predictive samples for ``y``. It must have the same shape as y plus an
        extra dimension at the end of size n_samples (chains and draws stacked). If str or
        None, ``idata`` must contain the posterior predictive group. If None, y_hat is taken
        equal to y, thus, y must be str too.
    log_weights: array or DataArray
        Smoothed log_weights. It must have the same shape as ``y_hat``
    dask_kwargs : dict, optional
        Dask related kwargs passed to :func:`~arviz.wrap_xarray_ufunc`.

    Returns
    -------
    loo_pit: array or DataArray
        Value of the LOO-PIT at each observed data point.

    See Also
    --------
    plot_loo_pit : Plot Leave-One-Out probability integral transformation (PIT) predictive checks.
    loo : Compute Pareto-smoothed importance sampling leave-one-out
          cross-validation (PSIS-LOO-CV).
    plot_elpd : Plot pointwise elpd differences between two or more models.
    plot_khat : Plot Pareto tail indices for diagnosing convergence.

    Examples
    --------
    Calculate LOO-PIT values using as test quantity the observed values themselves.

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data("centered_eight")
           ...: az.loo_pit(idata=data, y="obs")

    Calculate LOO-PIT values using as test quantity the square of the difference between
    each observation and `mu`. Both ``y`` and ``y_hat`` inputs will be array-like,
    but ``idata`` will still be passed in order to calculate the ``log_weights`` from
    there.

    .. ipython::

        In [1]: T = data.observed_data.obs - data.posterior.mu.median(dim=("chain", "draw"))
           ...: T_hat = data.posterior_predictive.obs - data.posterior.mu
           ...: T_hat = T_hat.stack(__sample__=("chain", "draw"))
           ...: az.loo_pit(idata=data, y=T**2, y_hat=T_hat**2)

    """
    y_str = ''
    if idata is not None and (not isinstance(idata, InferenceData)):
        raise ValueError('idata must be of type InferenceData or None')
    if idata is None:
        if not all((isinstance(arg, (np.ndarray, xr.DataArray)) for arg in (y, y_hat, log_weights))):
            raise ValueError(f'all 3 y, y_hat and log_weights must be array or DataArray when idata is None but they are of types {[type(arg) for arg in (y, y_hat, log_weights)]}')
    else:
        if y_hat is None and isinstance(y, str):
            y_hat = y
        elif y_hat is None:
            raise ValueError('y_hat cannot be None if y is not a str')
        if isinstance(y, str):
            y_str = y
            y = idata.observed_data[y].values
        elif not isinstance(y, (np.ndarray, xr.DataArray)):
            raise ValueError(f'y must be of types array, DataArray or str, not {type(y)}')
        if isinstance(y_hat, str):
            y_hat = idata.posterior_predictive[y_hat].stack(__sample__=('chain', 'draw')).values
        elif not isinstance(y_hat, (np.ndarray, xr.DataArray)):
            raise ValueError(f'y_hat must be of types array, DataArray or str, not {type(y_hat)}')
        if log_weights is None:
            if y_str:
                try:
                    log_likelihood = _get_log_likelihood(idata, var_name=y_str)
                except TypeError:
                    log_likelihood = _get_log_likelihood(idata)
            else:
                log_likelihood = _get_log_likelihood(idata)
            log_likelihood = log_likelihood.stack(__sample__=('chain', 'draw'))
            posterior = convert_to_dataset(idata, group='posterior')
            n_chains = len(posterior.chain)
            n_samples = len(log_likelihood.__sample__)
            ess_p = ess(posterior, method='mean')
            reff = np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean() / n_samples if n_chains > 1 else 1
            log_weights = psislw(-log_likelihood, reff=reff)[0].values
        elif not isinstance(log_weights, (np.ndarray, xr.DataArray)):
            raise ValueError(f'log_weights must be None or of types array or DataArray, not {type(log_weights)}')
    if len(y.shape) + 1 != len(y_hat.shape):
        raise ValueError(f'y_hat must have 1 more dimension than y, but y_hat has {len(y_hat.shape)} dims and y has {len(y.shape)} dims')
    if y.shape != y_hat.shape[:-1]:
        raise ValueError(f'y has shape: {y.shape} which should be equal to y_hat shape (omitting the last dimension): {y_hat.shape}')
    if y_hat.shape != log_weights.shape:
        raise ValueError(f'y_hat and log_weights must have the same shape but have shapes {(y_hat.shape,)} and {log_weights.shape}')
    kwargs = {'input_core_dims': [[], ['__sample__'], ['__sample__']], 'output_core_dims': [[]], 'join': 'left'}
    ufunc_kwargs = {'n_dims': 1}
    if y.dtype.kind == 'i' or y_hat.dtype.kind == 'i':
        y, y_hat = smooth_data(y, y_hat)
    return _wrap_xarray_ufunc(_loo_pit, y, y_hat, log_weights, ufunc_kwargs=ufunc_kwargs, **kwargs)