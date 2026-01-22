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
@pytest.mark.parametrize('freq', ['ns', 'us', '1000us'])
def test_dt_round_nonnano_higher_resolution_no_op(self, freq):
    ser = Series(['2020-05-31 08:00:00', '2000-12-31 04:00:05', '1800-03-14 07:30:20'], dtype='datetime64[ms]')
    expected = ser.copy()
    result = ser.dt.round(freq)
    tm.assert_series_equal(result, expected)
    assert not np.shares_memory(ser.array._ndarray, result.array._ndarray)