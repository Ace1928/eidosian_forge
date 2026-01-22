from __future__ import annotations
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from .._utils import get_valid_kwargs
from ..exceptions import PlotnineError, PlotnineWarning
def lm_formula(data, xseq, **params):
    """
    Fit OLS / WLS using a formula
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    eval_env = _to_patsy_env(params['enviroment'])
    formula = params['formula']
    weights = data.get('weight', None)
    if weights is None:
        init_kwargs, fit_kwargs = separate_method_kwargs(params['method_args'], sm.OLS, sm.OLS.fit)
        model = smf.ols(formula, data, eval_env=eval_env, **init_kwargs)
    else:
        if np.any(weights < 0):
            raise ValueError('All weights must be greater than zero.')
        init_kwargs, fit_kwargs = separate_method_kwargs(params['method_args'], sm.OLS, sm.OLS.fit)
        model = smf.wls(formula, data, weights=weights, eval_env=eval_env, **init_kwargs)
    results = model.fit(**fit_kwargs)
    data = pd.DataFrame({'x': xseq})
    data['y'] = results.predict(data)
    if params['se']:
        from patsy import dmatrices
        _, predictors = dmatrices(formula, data, eval_env=eval_env)
        alpha = 1 - params['level']
        prstd, iv_l, iv_u = wls_prediction_std(results, predictors, alpha=alpha)
        data['se'] = prstd
        data['ymin'] = iv_l
        data['ymax'] = iv_u
    return data