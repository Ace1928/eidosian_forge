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
def test_dt_round_tz(self):
    ser = Series(pd.to_datetime(['2012-01-01 13:00:00', '2012-01-01 12:01:00', '2012-01-01 08:00:00']), name='xxx')
    result = ser.dt.tz_localize('UTC').dt.tz_convert('US/Eastern').dt.round('D')
    exp_values = pd.to_datetime(['2012-01-01', '2012-01-01', '2012-01-01']).tz_localize('US/Eastern')
    expected = Series(exp_values, name='xxx')
    tm.assert_series_equal(result, expected)