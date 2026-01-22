from datetime import datetime
import warnings
import dateutil
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import _get_period_range_edges
from pandas.tseries import offsets
@pytest.mark.parametrize('tz', [pytz.timezone('America/Los_Angeles'), dateutil.tz.gettz('America/Los_Angeles')])
def test_with_local_timezone(self, tz):
    local_timezone = tz
    start = datetime(year=2013, month=11, day=1, hour=0, minute=0, tzinfo=pytz.utc)
    end = datetime(year=2013, month=11, day=2, hour=0, minute=0, tzinfo=pytz.utc)
    index = date_range(start, end, freq='h', name='idx')
    series = Series(1, index=index)
    series = series.tz_convert(local_timezone)
    msg = "The 'kind' keyword in Series.resample is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = series.resample('D', kind='period').mean()
    expected_index = period_range(start=start, end=end, freq='D', name='idx') - offsets.Day()
    expected = Series(1.0, index=expected_index)
    tm.assert_series_equal(result, expected)