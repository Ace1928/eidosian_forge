import calendar
from datetime import (
import locale
import unicodedata
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas.errors import SettingWithCopyError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_valid_dt_with_missing_values(self):
    ser = Series(date_range('20130101', periods=5, freq='D'))
    ser.iloc[2] = pd.NaT
    for attr in ['microsecond', 'nanosecond', 'second', 'minute', 'hour', 'day']:
        expected = getattr(ser.dt, attr).copy()
        expected.iloc[2] = np.nan
        result = getattr(ser.dt, attr)
        tm.assert_series_equal(result, expected)
    result = ser.dt.date
    expected = Series([date(2013, 1, 1), date(2013, 1, 2), pd.NaT, date(2013, 1, 4), date(2013, 1, 5)], dtype='object')
    tm.assert_series_equal(result, expected)
    result = ser.dt.time
    expected = Series([time(0), time(0), pd.NaT, time(0), time(0)], dtype='object')
    tm.assert_series_equal(result, expected)