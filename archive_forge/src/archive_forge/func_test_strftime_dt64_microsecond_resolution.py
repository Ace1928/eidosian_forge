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
def test_strftime_dt64_microsecond_resolution(self):
    ser = Series([datetime(2013, 1, 1, 2, 32, 59), datetime(2013, 1, 2, 14, 32, 1)])
    result = ser.dt.strftime('%Y-%m-%d %H:%M:%S')
    expected = Series(['2013-01-01 02:32:59', '2013-01-02 14:32:01'])
    tm.assert_series_equal(result, expected)