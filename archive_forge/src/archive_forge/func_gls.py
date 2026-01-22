from __future__ import annotations
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from .._utils import get_valid_kwargs
from ..exceptions import PlotnineError, PlotnineWarning
def gls(data, xseq, **params):
    """
    Fit GLS
    """
    import statsmodels.api as sm
    if params['formula']:
        return gls_formula(data, xseq, **params)
    X = sm.add_constant(data['x'])
    Xseq = sm.add_constant(xseq)
    init_kwargs, fit_kwargs = separate_method_kwargs(params['method_args'], sm.OLS, sm.OLS.fit)
    model = sm.GLS(data['y'], X, **init_kwargs)
    results = model.fit(**fit_kwargs)
    data = pd.DataFrame({'x': xseq})
    data['y'] = results.predict(Xseq)
    if params['se']:
        alpha = 1 - params['level']
        prstd, iv_l, iv_u = wls_prediction_std(results, Xseq, alpha=alpha)
        data['se'] = prstd
        data['ymin'] = iv_l
        data['ymax'] = iv_u
    return data