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
def test_resample_with_dst_time_change(self):
    index = pd.DatetimeIndex([1457537600000000000, 1458059600000000000]).tz_localize('UTC').tz_convert('America/Chicago')
    df = DataFrame([1, 2], index=index)
    result = df.resample('12h', closed='right', label='right').last().ffill()
    expected_index_values = ['2016-03-09 12:00:00-06:00', '2016-03-10 00:00:00-06:00', '2016-03-10 12:00:00-06:00', '2016-03-11 00:00:00-06:00', '2016-03-11 12:00:00-06:00', '2016-03-12 00:00:00-06:00', '2016-03-12 12:00:00-06:00', '2016-03-13 00:00:00-06:00', '2016-03-13 13:00:00-05:00', '2016-03-14 01:00:00-05:00', '2016-03-14 13:00:00-05:00', '2016-03-15 01:00:00-05:00', '2016-03-15 13:00:00-05:00']
    index = pd.to_datetime(expected_index_values, utc=True).tz_convert('America/Chicago').as_unit(index.unit)
    index = pd.DatetimeIndex(index, freq='12h')
    expected = DataFrame([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0], index=index)
    tm.assert_frame_equal(result, expected)