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
def test_resample_nonexistent_time_bin_edge(self):
    index = date_range('2017-03-12', '2017-03-12 1:45:00', freq='15min')
    s = Series(np.zeros(len(index)), index=index)
    expected = s.tz_localize('US/Pacific')
    expected.index = pd.DatetimeIndex(expected.index, freq='900s')
    result = expected.resample('900s').mean()
    tm.assert_series_equal(result, expected)