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
def test_dt_tz_localize_categorical(self, tz_aware_fixture):
    tz = tz_aware_fixture
    datetimes = Series(['2019-01-01', '2019-01-01', '2019-01-02'], dtype='datetime64[ns]')
    categorical = datetimes.astype('category')
    result = categorical.dt.tz_localize(tz)
    expected = datetimes.dt.tz_localize(tz)
    tm.assert_series_equal(result, expected)