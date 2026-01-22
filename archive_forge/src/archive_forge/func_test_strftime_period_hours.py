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
def test_strftime_period_hours(self):
    ser = Series(period_range('20130101', periods=4, freq='h'))
    result = ser.dt.strftime('%Y/%m/%d %H:%M:%S')
    expected = Series(['2013/01/01 00:00:00', '2013/01/01 01:00:00', '2013/01/01 02:00:00', '2013/01/01 03:00:00'])
    tm.assert_series_equal(result, expected)