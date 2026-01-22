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
@pytest.mark.parametrize('method, dates', [['round', ['2012-01-02', '2012-01-02', '2012-01-01']], ['floor', ['2012-01-01', '2012-01-01', '2012-01-01']], ['ceil', ['2012-01-02', '2012-01-02', '2012-01-02']]])
def test_dt_round(self, method, dates):
    ser = Series(pd.to_datetime(['2012-01-01 13:00:00', '2012-01-01 12:01:00', '2012-01-01 08:00:00']), name='xxx')
    result = getattr(ser.dt, method)('D')
    expected = Series(pd.to_datetime(dates), name='xxx')
    tm.assert_series_equal(result, expected)