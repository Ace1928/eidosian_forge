import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('how', ['method', 'agg', 'transform'])
def test_groupby_raises_string(how, by, groupby_series, groupby_func, df_with_string_col):
    df = df_with_string_col
    args = get_groupby_method_args(groupby_func, df)
    gb = df.groupby(by=by)
    if groupby_series:
        gb = gb['d']
        if groupby_func == 'corrwith':
            assert not hasattr(gb, 'corrwith')
            return
    klass, msg = {'all': (None, ''), 'any': (None, ''), 'bfill': (None, ''), 'corrwith': (TypeError, 'Could not convert'), 'count': (None, ''), 'cumcount': (None, ''), 'cummax': ((NotImplementedError, TypeError), '(function|cummax) is not (implemented|supported) for (this|object) dtype'), 'cummin': ((NotImplementedError, TypeError), '(function|cummin) is not (implemented|supported) for (this|object) dtype'), 'cumprod': ((NotImplementedError, TypeError), '(function|cumprod) is not (implemented|supported) for (this|object) dtype'), 'cumsum': ((NotImplementedError, TypeError), '(function|cumsum) is not (implemented|supported) for (this|object) dtype'), 'diff': (TypeError, 'unsupported operand type'), 'ffill': (None, ''), 'fillna': (None, ''), 'first': (None, ''), 'idxmax': (None, ''), 'idxmin': (None, ''), 'last': (None, ''), 'max': (None, ''), 'mean': (TypeError, re.escape('agg function failed [how->mean,dtype->object]')), 'median': (TypeError, re.escape('agg function failed [how->median,dtype->object]')), 'min': (None, ''), 'ngroup': (None, ''), 'nunique': (None, ''), 'pct_change': (TypeError, 'unsupported operand type'), 'prod': (TypeError, re.escape('agg function failed [how->prod,dtype->object]')), 'quantile': (TypeError, "cannot be performed against 'object' dtypes!"), 'rank': (None, ''), 'sem': (ValueError, 'could not convert string to float'), 'shift': (None, ''), 'size': (None, ''), 'skew': (ValueError, 'could not convert string to float'), 'std': (ValueError, 'could not convert string to float'), 'sum': (None, ''), 'var': (TypeError, re.escape('agg function failed [how->var,dtype->'))}[groupby_func]
    if groupby_func == 'fillna':
        kind = 'Series' if groupby_series else 'DataFrame'
        warn_msg = f'{kind}GroupBy.fillna is deprecated'
    else:
        warn_msg = ''
    _call_and_check(klass, msg, how, gb, groupby_func, args, warn_msg)