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
def test_dt_accessor_limited_display_api(self):
    ser = Series(date_range('20130101', periods=5, freq='D'), name='xxx')
    results = get_dir(ser)
    tm.assert_almost_equal(results, sorted(set(ok_for_dt + ok_for_dt_methods)))
    ser = Series(date_range('2015-01-01', '2016-01-01', freq='min'), name='xxx')
    ser = ser.dt.tz_localize('UTC').dt.tz_convert('America/Chicago')
    results = get_dir(ser)
    tm.assert_almost_equal(results, sorted(set(ok_for_dt + ok_for_dt_methods)))
    idx = period_range('20130101', periods=5, freq='D', name='xxx').astype(object)
    with tm.assert_produces_warning(FutureWarning, match='Dtype inference'):
        ser = Series(idx)
    results = get_dir(ser)
    tm.assert_almost_equal(results, sorted(set(ok_for_period + ok_for_period_methods)))