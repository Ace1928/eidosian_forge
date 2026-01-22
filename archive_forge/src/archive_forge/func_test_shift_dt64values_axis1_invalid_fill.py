import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('as_cat', [pytest.param(True, marks=pytest.mark.xfail(reason='_can_hold_element incorrectly always returns True')), False])
@pytest.mark.parametrize('vals', [date_range('2020-01-01', periods=2), date_range('2020-01-01', periods=2, tz='US/Pacific'), pd.period_range('2020-01-01', periods=2, freq='D'), pd.timedelta_range('2020 Days', periods=2, freq='D'), pd.interval_range(0, 3, periods=2), pytest.param(pd.array([1, 2], dtype='Int64'), marks=pytest.mark.xfail(reason='_can_hold_element incorrectly always returns True')), pytest.param(pd.array([1, 2], dtype='Float32'), marks=pytest.mark.xfail(reason='_can_hold_element incorrectly always returns True'))], ids=lambda x: str(x.dtype))
def test_shift_dt64values_axis1_invalid_fill(self, vals, as_cat):
    ser = Series(vals)
    if as_cat:
        ser = ser.astype('category')
    df = DataFrame({'A': ser})
    result = df.shift(-1, axis=1, fill_value='foo')
    expected = DataFrame({'A': ['foo', 'foo']})
    tm.assert_frame_equal(result, expected)
    df2 = DataFrame({'A': ser, 'B': ser})
    df2._consolidate_inplace()
    result = df2.shift(-1, axis=1, fill_value='foo')
    expected = DataFrame({'A': df2['B'], 'B': ['foo', 'foo']})
    tm.assert_frame_equal(result, expected)
    df3 = DataFrame({'A': ser})
    df3['B'] = ser
    assert len(df3._mgr.arrays) == 2
    result = df3.shift(-1, axis=1, fill_value='foo')
    tm.assert_frame_equal(result, expected)