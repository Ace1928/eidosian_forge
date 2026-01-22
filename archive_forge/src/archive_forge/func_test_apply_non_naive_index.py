from textwrap import dedent
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_apply_non_naive_index():

    def weighted_quantile(series, weights, q):
        series = series.sort_values()
        cumsum = weights.reindex(series.index).fillna(0).cumsum()
        cutoff = cumsum.iloc[-1] * q
        return series[cumsum >= cutoff].iloc[0]
    times = date_range('2017-6-23 18:00', periods=8, freq='15min', tz='UTC')
    data = Series([1.0, 1, 1, 1, 1, 2, 2, 0], index=times)
    weights = Series([160.0, 91, 65, 43, 24, 10, 1, 0], index=times)
    result = data.resample('D').apply(weighted_quantile, weights=weights, q=0.5)
    ind = date_range('2017-06-23 00:00:00+00:00', '2017-06-23 00:00:00+00:00', freq='D', tz='UTC')
    expected = Series([1.0], index=ind)
    tm.assert_series_equal(result, expected)