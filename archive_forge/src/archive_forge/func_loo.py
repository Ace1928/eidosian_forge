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
def loo(data, pointwise=None, var_name=None, reff=None, scale=None):
    """Compute Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO-CV).

    Estimates the expected log pointwise predictive density (elpd) using Pareto-smoothed
    importance sampling leave-one-out cross-validation (PSIS-LOO-CV). Also calculates LOO's
    standard error and the effective number of parameters. Read more theory here
    https://arxiv.org/abs/1507.04544 and here https://arxiv.org/abs/1507.02646

    Parameters
    ----------
    data: obj
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Refer to documentation of
        :func:`arviz.convert_to_dataset` for details.
    pointwise: bool, optional
        If True the pointwise predictive accuracy will be returned. Defaults to
        ``stats.ic_pointwise`` rcParam.
    var_name : str, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for loo computation.
    reff: float, optional
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples. Computed from trace by default.
    scale: str
        Output scale for loo. Available options are:

        - ``log`` : (default) log-score
        - ``negative_log`` : -1 * log-score
        - ``deviance`` : -2 * log-score

        A higher log-score (or a lower deviance or negative log_score) indicates a model with
        better predictive accuracy.

    Returns
    -------
    ELPDData object (inherits from :class:`pandas.Series`) with the following row/attributes:
    elpd: approximated expected log pointwise predictive density (elpd)
    se: standard error of the elpd
    p_loo: effective number of parameters
    shape_warn: bool
        True if the estimated shape parameter of
        Pareto distribution is greater than 0.7 for one or more samples
    loo_i: array of pointwise predictive accuracy, only if pointwise True
    pareto_k: array of Pareto shape values, only if pointwise True
    scale: scale of the elpd

        The returned object has a custom print method that overrides pd.Series method.

    See Also
    --------
    compare : Compare models based on PSIS-LOO loo or WAIC waic cross-validation.
    waic : Compute the widely applicable information criterion.
    plot_compare : Summary plot for model comparison.
    plot_elpd : Plot pointwise elpd differences between two or more models.
    plot_khat : Plot Pareto tail indices for diagnosing convergence.

    Examples
    --------
    Calculate LOO of a model:

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data("centered_eight")
           ...: az.loo(data)

    Calculate LOO of a model and return the pointwise values:

    .. ipython::

        In [2]: data_loo = az.loo(data, pointwise=True)
           ...: data_loo.loo_i
    """
    inference_data = convert_to_inference_data(data)
    log_likelihood = _get_log_likelihood(inference_data, var_name=var_name)
    pointwise = rcParams['stats.ic_pointwise'] if pointwise is None else pointwise
    log_likelihood = log_likelihood.stack(__sample__=('chain', 'draw'))
    shape = log_likelihood.shape
    n_samples = shape[-1]
    n_data_points = np.prod(shape[:-1])
    scale = rcParams['stats.ic_scale'] if scale is None else scale.lower()
    if scale == 'deviance':
        scale_value = -2
    elif scale == 'log':
        scale_value = 1
    elif scale == 'negative_log':
        scale_value = -1
    else:
        raise TypeError('Valid scale values are "deviance", "log", "negative_log"')
    if reff is None:
        if not hasattr(inference_data, 'posterior'):
            raise TypeError('Must be able to extract a posterior group from data.')
        posterior = inference_data.posterior
        n_chains = len(posterior.chain)
        if n_chains == 1:
            reff = 1.0
        else:
            ess_p = ess(posterior, method='mean')
            reff = np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean() / n_samples
    log_weights, pareto_shape = psislw(-log_likelihood, reff)
    log_weights += log_likelihood
    warn_mg = False
    if np.any(pareto_shape > 0.7):
        warnings.warn('Estimated shape parameter of Pareto distribution is greater than 0.7 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.')
        warn_mg = True
    ufunc_kwargs = {'n_dims': 1, 'ravel': False}
    kwargs = {'input_core_dims': [['__sample__']]}
    loo_lppd_i = scale_value * _wrap_xarray_ufunc(_logsumexp, log_weights, ufunc_kwargs=ufunc_kwargs, **kwargs)
    loo_lppd = loo_lppd_i.values.sum()
    loo_lppd_se = (n_data_points * np.var(loo_lppd_i.values)) ** 0.5
    lppd = np.sum(_wrap_xarray_ufunc(_logsumexp, log_likelihood, func_kwargs={'b_inv': n_samples}, ufunc_kwargs=ufunc_kwargs, **kwargs).values)
    p_loo = lppd - loo_lppd / scale_value
    if not pointwise:
        return ELPDData(data=[loo_lppd, loo_lppd_se, p_loo, n_samples, n_data_points, warn_mg, scale], index=['elpd_loo', 'se', 'p_loo', 'n_samples', 'n_data_points', 'warning', 'scale'])
    if np.equal(loo_lppd, loo_lppd_i).all():
        warnings.warn('The point-wise LOO is the same with the sum LOO, please double check the Observed RV in your model to make sure it returns element-wise logp.')
    return ELPDData(data=[loo_lppd, loo_lppd_se, p_loo, n_samples, n_data_points, warn_mg, loo_lppd_i.rename('loo_i'), pareto_shape, scale], index=['elpd_loo', 'se', 'p_loo', 'n_samples', 'n_data_points', 'warning', 'loo_i', 'pareto_k', 'scale'])