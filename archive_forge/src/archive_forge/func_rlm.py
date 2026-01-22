from __future__ import annotations
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from .._utils import get_valid_kwargs
from ..exceptions import PlotnineError, PlotnineWarning
def rlm(data, xseq, **params):
    """
    Fit RLM
    """
    import statsmodels.api as sm
    if params['formula']:
        return rlm_formula(data, xseq, **params)
    X = sm.add_constant(data['x'])
    Xseq = sm.add_constant(xseq)
    init_kwargs, fit_kwargs = separate_method_kwargs(params['method_args'], sm.RLM, sm.RLM.fit)
    model = sm.RLM(data['y'], X, **init_kwargs)
    results = model.fit(**fit_kwargs)
    data = pd.DataFrame({'x': xseq})
    data['y'] = results.predict(Xseq)
    if params['se']:
        warnings.warn('Confidence intervals are not yet implemented for RLM smoothing.', PlotnineWarning)
    return data