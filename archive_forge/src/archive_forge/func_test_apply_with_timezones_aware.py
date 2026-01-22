from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_with_timezones_aware():
    dates = ['2001-01-01'] * 2 + ['2001-01-02'] * 2 + ['2001-01-03'] * 2
    index_no_tz = pd.DatetimeIndex(dates)
    index_tz = pd.DatetimeIndex(dates, tz='UTC')
    df1 = DataFrame({'x': list(range(2)) * 3, 'y': range(6), 't': index_no_tz})
    df2 = DataFrame({'x': list(range(2)) * 3, 'y': range(6), 't': index_tz})
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result1 = df1.groupby('x', group_keys=False).apply(lambda df: df[['x', 'y']].copy())
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result2 = df2.groupby('x', group_keys=False).apply(lambda df: df[['x', 'y']].copy())
    tm.assert_frame_equal(result1, result2)