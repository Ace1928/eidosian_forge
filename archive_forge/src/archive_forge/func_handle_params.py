from statsmodels.compat.pandas import is_int_index
import contextlib
import warnings
import datetime as dt
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tools.tools import pinv_extended, Bunch
from statsmodels.tools.sm_exceptions import PrecisionWarning, ValueWarning
from statsmodels.tools.numdiff import (_get_epsilon, approx_hess_cs,
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, aicc, bic, hqic
import statsmodels.base.wrapper as wrap
import statsmodels.tsa.base.prediction as pred
from statsmodels.base.data import PandasData
import statsmodels.tsa.base.tsa_model as tsbase
from .news import NewsResults
from .simulation_smoother import SimulationSmoother
from .kalman_smoother import SmootherResults
from .kalman_filter import INVERT_UNIVARIATE, SOLVE_LU, MEMORY_CONSERVE
from .initialization import Initialization
from .tools import prepare_exog, concat, _safe_cond, get_impact_dates
def handle_params(self, params, transformed=True, includes_fixed=False, return_jacobian=False):
    """
        Ensure model parameters satisfy shape and other requirements
        """
    params = np.array(params, ndmin=1)
    if np.issubdtype(params.dtype, np.integer):
        params = params.astype(np.float64)
    if not includes_fixed and self._has_fixed_params:
        k_params = len(self.param_names)
        new_params = np.zeros(k_params, dtype=params.dtype) * np.nan
        new_params[self._free_params_index] = params
        params = new_params
    if not transformed:
        if not includes_fixed and self._has_fixed_params:
            params[self._fixed_params_index] = list(self._fixed_params.values())
        if return_jacobian:
            transform_score = self.transform_jacobian(params)
        params = self.transform_params(params)
    if not includes_fixed and self._has_fixed_params:
        params[self._fixed_params_index] = list(self._fixed_params.values())
    return (params, transform_score) if return_jacobian else params