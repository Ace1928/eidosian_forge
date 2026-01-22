from textwrap import dedent
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_apply_with_mutated_index():
    index = date_range('1-1-2015', '12-31-15', freq='D')
    df = DataFrame(data={'col1': np.random.default_rng(2).random(len(index))}, index=index)

    def f(x):
        s = Series([1, 2], index=['a', 'b'])
        return s
    expected = df.groupby(pd.Grouper(freq='ME')).apply(f)
    result = df.resample('ME').apply(f)
    tm.assert_frame_equal(result, expected)
    expected = df['col1'].groupby(pd.Grouper(freq='ME'), group_keys=False).apply(f)
    result = df['col1'].resample('ME').apply(f)
    tm.assert_series_equal(result, expected)