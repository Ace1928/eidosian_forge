from datetime import (
import re
from dateutil.tz import (
import numpy as np
import pytest
import pytz
from pandas._libs import index as libindex
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_getitem_setitem_datetimeindex():
    N = 50
    rng = date_range('1/1/1990', periods=N, freq='h', tz='US/Eastern')
    ts = Series(np.random.default_rng(2).standard_normal(N), index=rng)
    result = ts['1990-01-01 04:00:00']
    expected = ts.iloc[4]
    assert result == expected
    result = ts.copy()
    result['1990-01-01 04:00:00'] = 0
    result['1990-01-01 04:00:00'] = ts.iloc[4]
    tm.assert_series_equal(result, ts)
    result = ts['1990-01-01 04:00:00':'1990-01-01 07:00:00']
    expected = ts[4:8]
    tm.assert_series_equal(result, expected)
    result = ts.copy()
    result['1990-01-01 04:00:00':'1990-01-01 07:00:00'] = 0
    result['1990-01-01 04:00:00':'1990-01-01 07:00:00'] = ts[4:8]
    tm.assert_series_equal(result, ts)
    lb = '1990-01-01 04:00:00'
    rb = '1990-01-01 07:00:00'
    result = ts[(ts.index >= lb) & (ts.index <= rb)]
    expected = ts[4:8]
    tm.assert_series_equal(result, expected)
    lb = '1990-01-01 04:00:00-0500'
    rb = '1990-01-01 07:00:00-0500'
    result = ts[(ts.index >= lb) & (ts.index <= rb)]
    expected = ts[4:8]
    tm.assert_series_equal(result, expected)
    msg = 'Cannot compare tz-naive and tz-aware datetime-like objects'
    naive = datetime(1990, 1, 1, 4)
    for key in [naive, Timestamp(naive), np.datetime64(naive, 'ns')]:
        with pytest.raises(KeyError, match=re.escape(repr(key))):
            ts[key]
    result = ts.copy()
    result[naive] = ts.iloc[4]
    assert result.index.dtype == object
    tm.assert_index_equal(result.index[:-1], rng.astype(object))
    assert result.index[-1] == naive
    msg = 'Cannot compare tz-naive and tz-aware datetime-like objects'
    with pytest.raises(TypeError, match=msg):
        ts[naive:datetime(1990, 1, 1, 7)]
    result = ts.copy()
    with pytest.raises(TypeError, match=msg):
        result[naive:datetime(1990, 1, 1, 7)] = 0
    with pytest.raises(TypeError, match=msg):
        result[naive:datetime(1990, 1, 1, 7)] = 99
    tm.assert_series_equal(result, ts)
    lb = naive
    rb = datetime(1990, 1, 1, 7)
    msg = 'Invalid comparison between dtype=datetime64\\[ns, US/Eastern\\] and datetime'
    with pytest.raises(TypeError, match=msg):
        ts[(ts.index >= lb) & (ts.index <= rb)]
    lb = Timestamp(naive).tz_localize(rng.tzinfo)
    rb = Timestamp(datetime(1990, 1, 1, 7)).tz_localize(rng.tzinfo)
    result = ts[(ts.index >= lb) & (ts.index <= rb)]
    expected = ts[4:8]
    tm.assert_series_equal(result, expected)
    result = ts[ts.index[4]]
    expected = ts.iloc[4]
    assert result == expected
    result = ts[ts.index[4:8]]
    expected = ts[4:8]
    tm.assert_series_equal(result, expected)
    result = ts.copy()
    result[ts.index[4:8]] = 0
    result.iloc[4:8] = ts.iloc[4:8]
    tm.assert_series_equal(result, ts)
    result = ts['1990-01-02']
    expected = ts[24:48]
    tm.assert_series_equal(result, expected)
    result = ts.copy()
    result['1990-01-02'] = 0
    result['1990-01-02'] = ts[24:48]
    tm.assert_series_equal(result, ts)