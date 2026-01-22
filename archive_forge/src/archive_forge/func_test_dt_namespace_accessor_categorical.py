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
def test_dt_namespace_accessor_categorical(self):
    dti = DatetimeIndex(['20171111', '20181212']).repeat(2)
    ser = Series(pd.Categorical(dti), name='foo')
    result = ser.dt.year
    expected = Series([2017, 2017, 2018, 2018], dtype='int32', name='foo')
    tm.assert_series_equal(result, expected)