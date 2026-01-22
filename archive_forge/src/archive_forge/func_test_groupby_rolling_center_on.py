import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
def test_groupby_rolling_center_on(self):
    df = DataFrame(data={'Date': date_range('2020-01-01', '2020-01-10'), 'gb': ['group_1'] * 6 + ['group_2'] * 4, 'value': range(10)})
    result = df.groupby('gb').rolling(6, on='Date', center=True, min_periods=1).value.mean()
    mi = MultiIndex.from_arrays([df['gb'], df['Date']], names=['gb', 'Date'])
    expected = Series([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 7.0, 7.5, 7.5, 7.5], name='value', index=mi)
    tm.assert_series_equal(result, expected)