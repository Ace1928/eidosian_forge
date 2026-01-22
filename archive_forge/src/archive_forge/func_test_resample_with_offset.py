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
@pytest.mark.parametrize('start,end,start_freq,end_freq,offset', [('19910905', '19910909 03:00', 'h', '24h', '10h'), ('19910905', '19910909 12:00', 'h', '24h', '10h'), ('19910905', '19910909 23:00', 'h', '24h', '10h'), ('19910905 10:00', '19910909', 'h', '24h', '10h'), ('19910905 10:00', '19910909 10:00', 'h', '24h', '10h'), ('19910905', '19910909 10:00', 'h', '24h', '10h'), ('19910905 12:00', '19910909', 'h', '24h', '10h'), ('19910905 12:00', '19910909 03:00', 'h', '24h', '10h'), ('19910905 12:00', '19910909 12:00', 'h', '24h', '10h'), ('19910905 12:00', '19910909 12:00', 'h', '24h', '34h'), ('19910905 12:00', '19910909 12:00', 'h', '17h', '10h'), ('19910905 12:00', '19910909 12:00', 'h', '17h', '3h'), ('19910905', '19910913 06:00', '2h', '24h', '10h'), ('19910905', '19910905 01:39', 'Min', '5Min', '3Min'), ('19910905', '19910905 03:18', '2Min', '5Min', '3Min')])
def test_resample_with_offset(self, start, end, start_freq, end_freq, offset):
    pi = period_range(start, end, freq=start_freq)
    ser = Series(np.arange(len(pi)), index=pi)
    msg = 'Resampling with a PeriodIndex is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs = ser.resample(end_freq, offset=offset)
    result = rs.mean()
    result = result.to_timestamp(end_freq)
    expected = ser.to_timestamp().resample(end_freq, offset=offset).mean()
    tm.assert_series_equal(result, expected)