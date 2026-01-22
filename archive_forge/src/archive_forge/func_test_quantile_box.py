import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_quantile_box(self, interp_method, request, using_array_manager):
    interpolation, method = interp_method
    if method == 'table' and using_array_manager:
        request.applymarker(pytest.mark.xfail(reason='Axis name incorrectly set.'))
    df = DataFrame({'A': [Timestamp('2011-01-01'), Timestamp('2011-01-02'), Timestamp('2011-01-03')], 'B': [Timestamp('2011-01-01', tz='US/Eastern'), Timestamp('2011-01-02', tz='US/Eastern'), Timestamp('2011-01-03', tz='US/Eastern')], 'C': [pd.Timedelta('1 days'), pd.Timedelta('2 days'), pd.Timedelta('3 days')]})
    res = df.quantile(0.5, numeric_only=False, interpolation=interpolation, method=method)
    exp = Series([Timestamp('2011-01-02'), Timestamp('2011-01-02', tz='US/Eastern'), pd.Timedelta('2 days')], name=0.5, index=['A', 'B', 'C'])
    tm.assert_series_equal(res, exp)
    res = df.quantile([0.5], numeric_only=False, interpolation=interpolation, method=method)
    exp = DataFrame([[Timestamp('2011-01-02'), Timestamp('2011-01-02', tz='US/Eastern'), pd.Timedelta('2 days')]], index=[0.5], columns=['A', 'B', 'C'])
    tm.assert_frame_equal(res, exp)