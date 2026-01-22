import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('mix_groupings', [True, False])
@pytest.mark.parametrize('as_series', [True, False])
@pytest.mark.parametrize('val1,val2', [('foo', 'bar'), (1, 2), (1.0, 2.0)])
@pytest.mark.parametrize('fill_method,limit,exp_vals', [('ffill', None, [np.nan, np.nan, 'val1', 'val1', 'val1', 'val2', 'val2', 'val2']), ('ffill', 1, [np.nan, np.nan, 'val1', 'val1', np.nan, 'val2', 'val2', np.nan]), ('bfill', None, ['val1', 'val1', 'val1', 'val2', 'val2', 'val2', np.nan, np.nan]), ('bfill', 1, [np.nan, 'val1', 'val1', np.nan, 'val2', 'val2', np.nan, np.nan])])
def test_group_fill_methods(mix_groupings, as_series, val1, val2, fill_method, limit, exp_vals):
    vals = [np.nan, np.nan, val1, np.nan, np.nan, val2, np.nan, np.nan]
    _exp_vals = list(exp_vals)
    for index, exp_val in enumerate(_exp_vals):
        if exp_val == 'val1':
            _exp_vals[index] = val1
        elif exp_val == 'val2':
            _exp_vals[index] = val2
    if mix_groupings:
        keys = ['a', 'b'] * len(vals)

        def interweave(list_obj):
            temp = []
            for x in list_obj:
                temp.extend([x, x])
            return temp
        _exp_vals = interweave(_exp_vals)
        vals = interweave(vals)
    else:
        keys = ['a'] * len(vals) + ['b'] * len(vals)
        _exp_vals = _exp_vals * 2
        vals = vals * 2
    df = DataFrame({'key': keys, 'val': vals})
    if as_series:
        result = getattr(df.groupby('key')['val'], fill_method)(limit=limit)
        exp = Series(_exp_vals, name='val')
        tm.assert_series_equal(result, exp)
    else:
        result = getattr(df.groupby('key'), fill_method)(limit=limit)
        exp = DataFrame({'val': _exp_vals})
        tm.assert_frame_equal(result, exp)