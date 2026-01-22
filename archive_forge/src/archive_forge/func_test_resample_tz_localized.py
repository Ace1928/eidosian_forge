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
def test_resample_tz_localized(self, unit):
    dr = date_range(start='2012-4-13', end='2012-5-1', unit=unit)
    ts = Series(range(len(dr)), index=dr)
    ts_utc = ts.tz_localize('UTC')
    ts_local = ts_utc.tz_convert('America/Los_Angeles')
    result = ts_local.resample('W').mean()
    ts_local_naive = ts_local.copy()
    ts_local_naive.index = ts_local_naive.index.tz_localize(None)
    exp = ts_local_naive.resample('W').mean().tz_localize('America/Los_Angeles')
    exp.index = pd.DatetimeIndex(exp.index, freq='W')
    tm.assert_series_equal(result, exp)
    result = ts_local.resample('D').mean()