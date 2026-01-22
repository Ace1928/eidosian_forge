from statsmodels.compat.pandas import deprecate_kwarg
import os
import subprocess
import tempfile
import re
from warnings import warn
import pandas as pd
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (X13NotFoundError,
def pandas_to_series_spec(x):
    if hasattr(x, 'columns'):
        if len(x.columns) > 1:
            raise ValueError('Does not handle DataFrame with more than one column')
        x = x[x.columns[0]]
    data = '({})'.format('\n'.join(map(str, x.values.tolist())))
    try:
        period = _freq_to_period[x.index.freqstr]
    except (AttributeError, ValueError):
        from pandas.tseries.api import infer_freq
        period = _freq_to_period[infer_freq(x.index)]
    start_date = x.index[0]
    if period == 12:
        year, stperiod = (start_date.year, start_date.month)
    elif period == 4:
        year, stperiod = (start_date.year, start_date.quarter)
    else:
        raise ValueError('Only monthly and quarterly periods are supported. Please report or send a pull request if you want this extended.')
    if hasattr(x, 'name'):
        name = x.name or 'Unnamed Series'
    else:
        name = 'Unnamed Series'
    series_spec = SeriesSpec(data=data, name=name, period=period, title=name, start='{}.{}'.format(year, stperiod))
    return series_spec