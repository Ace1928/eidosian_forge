from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_datetimetz(self):
    df = DataFrame({'A': date_range('20130101', periods=3, tz='US/Eastern'), 'B': [0, np.nan, 2]})
    result = df.replace(np.nan, 1)
    expected = DataFrame({'A': date_range('20130101', periods=3, tz='US/Eastern'), 'B': Series([0, 1, 2], dtype='float64')})
    tm.assert_frame_equal(result, expected)
    result = df.fillna(1)
    tm.assert_frame_equal(result, expected)
    result = df.replace(0, np.nan)
    expected = DataFrame({'A': date_range('20130101', periods=3, tz='US/Eastern'), 'B': [np.nan, np.nan, 2]})
    tm.assert_frame_equal(result, expected)
    result = df.replace(Timestamp('20130102', tz='US/Eastern'), Timestamp('20130104', tz='US/Eastern'))
    expected = DataFrame({'A': [Timestamp('20130101', tz='US/Eastern'), Timestamp('20130104', tz='US/Eastern'), Timestamp('20130103', tz='US/Eastern')], 'B': [0, np.nan, 2]})
    expected['A'] = expected['A'].dt.as_unit('ns')
    tm.assert_frame_equal(result, expected)
    result = df.copy()
    result.iloc[1, 0] = np.nan
    result = result.replace({'A': pd.NaT}, Timestamp('20130104', tz='US/Eastern'))
    tm.assert_frame_equal(result, expected)
    result = df.copy()
    result.iloc[1, 0] = np.nan
    result = result.replace({'A': pd.NaT}, Timestamp('20130104', tz='US/Pacific'))
    expected = DataFrame({'A': [Timestamp('20130101', tz='US/Eastern'), Timestamp('20130104', tz='US/Pacific').tz_convert('US/Eastern'), Timestamp('20130103', tz='US/Eastern')], 'B': [0, np.nan, 2]})
    expected['A'] = expected['A'].dt.as_unit('ns')
    tm.assert_frame_equal(result, expected)
    result = df.copy()
    result.iloc[1, 0] = np.nan
    result = result.replace({'A': np.nan}, Timestamp('20130104'))
    expected = DataFrame({'A': [Timestamp('20130101', tz='US/Eastern'), Timestamp('20130104'), Timestamp('20130103', tz='US/Eastern')], 'B': [0, np.nan, 2]})
    tm.assert_frame_equal(result, expected)