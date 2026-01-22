import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_ragged_apply(engine_and_raw):
    engine, raw = engine_and_raw
    df = DataFrame({'B': range(5)})
    df.index = [Timestamp('20130101 09:00:00'), Timestamp('20130101 09:00:02'), Timestamp('20130101 09:00:03'), Timestamp('20130101 09:00:05'), Timestamp('20130101 09:00:06')]
    f = lambda x: 1
    result = df.rolling(window='1s', min_periods=1).apply(f, engine=engine, raw=raw)
    expected = df.copy()
    expected['B'] = 1.0
    tm.assert_frame_equal(result, expected)
    result = df.rolling(window='2s', min_periods=1).apply(f, engine=engine, raw=raw)
    expected = df.copy()
    expected['B'] = 1.0
    tm.assert_frame_equal(result, expected)
    result = df.rolling(window='5s', min_periods=1).apply(f, engine=engine, raw=raw)
    expected = df.copy()
    expected['B'] = 1.0
    tm.assert_frame_equal(result, expected)