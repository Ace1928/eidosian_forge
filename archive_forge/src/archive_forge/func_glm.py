from __future__ import annotations
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from .._utils import get_valid_kwargs
from ..exceptions import PlotnineError, PlotnineWarning
def glm(data, xseq, **params):
    """
    Fit GLM
    """
    import statsmodels.api as sm
    if params['formula']:
        return glm_formula(data, xseq, **params)
    X = sm.add_constant(data['x'])
    Xseq = sm.add_constant(xseq)
    init_kwargs, fit_kwargs = separate_method_kwargs(params['method_args'], sm.GLM, sm.GLM.fit)
    model = sm.GLM(data['y'], X, **init_kwargs)
    results = model.fit(**fit_kwargs)
    data = pd.DataFrame({'x': xseq})
    data['y'] = results.predict(Xseq)
    if params['se']:
        prediction = results.get_prediction(Xseq)
        ci = prediction.conf_int(1 - params['level'])
        data['ymin'] = ci[:, 0]
        data['ymax'] = ci[:, 1]
    return data