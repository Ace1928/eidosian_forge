from __future__ import annotations
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from .._utils import get_valid_kwargs
from ..exceptions import PlotnineError, PlotnineWarning
def mavg(data, xseq, **params):
    """
    Fit moving average
    """
    window = params['method_args']['window']
    rolling = data['y'].rolling(**params['method_args'])
    y = rolling.mean()[window:]
    n = len(data)
    stderr = rolling.std()[window:]
    x = data['x'][window:]
    data = pd.DataFrame({'x': x, 'y': y})
    data.reset_index(inplace=True, drop=True)
    if params['se']:
        dof = n - window
        data['ymin'], data['ymax'] = tdist_ci(y, dof, stderr, params['level'])
        data['se'] = stderr
    return data