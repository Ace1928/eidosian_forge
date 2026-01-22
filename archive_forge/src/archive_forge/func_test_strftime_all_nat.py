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
@pytest.mark.parametrize('data', [DatetimeIndex([pd.NaT]), PeriodIndex([pd.NaT], dtype='period[D]')])
def test_strftime_all_nat(self, data):
    ser = Series(data)
    with tm.assert_produces_warning(None):
        result = ser.dt.strftime('%Y-%m-%d')
    expected = Series([np.nan], dtype=object)
    tm.assert_series_equal(result, expected)