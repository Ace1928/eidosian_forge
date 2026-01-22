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
def test_dt_timetz_accessor(self, tz_naive_fixture):
    tz = maybe_get_tz(tz_naive_fixture)
    dtindex = DatetimeIndex(['2014-04-04 23:56', '2014-07-18 21:24', '2015-11-22 22:14'], tz=tz)
    ser = Series(dtindex)
    expected = Series([time(23, 56, tzinfo=tz), time(21, 24, tzinfo=tz), time(22, 14, tzinfo=tz)])
    result = ser.dt.timetz
    tm.assert_series_equal(result, expected)