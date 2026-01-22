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
def test_dt_accessor_ambiguous_freq_conversions(self):
    ser = Series(date_range('2015-01-01', '2016-01-01', freq='min'), name='xxx')
    ser = ser.dt.tz_localize('UTC').dt.tz_convert('America/Chicago')
    exp_values = date_range('2015-01-01', '2016-01-01', freq='min', tz='UTC').tz_convert('America/Chicago')
    exp_values = exp_values._with_freq(None)
    expected = Series(exp_values, name='xxx')
    tm.assert_series_equal(ser, expected)