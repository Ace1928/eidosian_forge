from textwrap import dedent
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_groupby_with_origin():
    freq = '1399min'
    start, end = ('1/1/2000 00:00:00', '1/31/2000 00:00')
    middle = '1/15/2000 00:00:00'
    rng = date_range(start, end, freq='1231min')
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    ts2 = ts[middle:end]
    simple_grouper = pd.Grouper(freq=freq)
    count_ts = ts.groupby(simple_grouper).agg('count')
    count_ts = count_ts[middle:end]
    count_ts2 = ts2.groupby(simple_grouper).agg('count')
    with pytest.raises(AssertionError, match='Index are different'):
        tm.assert_index_equal(count_ts.index, count_ts2.index)
    origin = Timestamp(0)
    adjusted_grouper = pd.Grouper(freq=freq, origin=origin)
    adjusted_count_ts = ts.groupby(adjusted_grouper).agg('count')
    adjusted_count_ts = adjusted_count_ts[middle:end]
    adjusted_count_ts2 = ts2.groupby(adjusted_grouper).agg('count')
    tm.assert_series_equal(adjusted_count_ts, adjusted_count_ts2)
    origin_future = Timestamp(0) + pd.Timedelta('1399min') * 30000
    adjusted_grouper2 = pd.Grouper(freq=freq, origin=origin_future)
    adjusted2_count_ts = ts.groupby(adjusted_grouper2).agg('count')
    adjusted2_count_ts = adjusted2_count_ts[middle:end]
    adjusted2_count_ts2 = ts2.groupby(adjusted_grouper2).agg('count')
    tm.assert_series_equal(adjusted2_count_ts, adjusted2_count_ts2)
    tm.assert_series_equal(adjusted_count_ts, adjusted2_count_ts2)