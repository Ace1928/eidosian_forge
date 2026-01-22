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
def test_upsample_daily_business_daily(self, simple_period_range_series):
    ts = simple_period_range_series('1/1/2000', '2/1/2000', freq='B')
    result = ts.resample('D').asfreq()
    expected = ts.asfreq('D').reindex(period_range('1/3/2000', '2/1/2000'))
    tm.assert_series_equal(result, expected)
    ts = simple_period_range_series('1/1/2000', '2/1/2000')
    msg = "The 'convention' keyword in Series.resample is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = ts.resample('h', convention='s').asfreq()
    exp_rng = period_range('1/1/2000', '2/1/2000 23:00', freq='h')
    expected = ts.asfreq('h', how='s').reindex(exp_rng)
    tm.assert_series_equal(result, expected)