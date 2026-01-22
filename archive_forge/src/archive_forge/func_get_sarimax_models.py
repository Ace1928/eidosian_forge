import numpy as np
import pandas as pd
from statsmodels.tools.tools import Bunch
from .results import results_varmax
from statsmodels.tsa.statespace import sarimax, varmax
from numpy.testing import assert_raises, assert_allclose
def get_sarimax_models(endog, filter_univariate=False, **kwargs):
    kwargs.setdefault('tolerance', 0)
    mod_conc = sarimax.SARIMAX(endog, **kwargs)
    mod_conc.ssm.filter_concentrated = True
    mod_conc.ssm.filter_univariate = filter_univariate
    params_conc = mod_conc.start_params
    params_conc[-1] = 1
    res_conc = mod_conc.smooth(params_conc)
    scale = res_conc.scale
    mod_orig = sarimax.SARIMAX(endog, **kwargs)
    mod_orig.ssm.filter_univariate = filter_univariate
    params_orig = params_conc.copy()
    k_vars = 1 + kwargs.get('measurement_error', False)
    params_orig[-k_vars:] = scale * params_conc[-k_vars:]
    res_orig = mod_orig.smooth(params_orig)
    return Bunch(**{'mod_conc': mod_conc, 'params_conc': params_conc, 'mod_orig': mod_orig, 'params_orig': params_orig, 'res_conc': res_conc, 'res_orig': res_orig, 'scale': scale})