from datetime import datetime
from functools import partial
import numpy as np
import pytest
import pytz
from pandas._libs import lib
from pandas._typing import DatetimeNaTType
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import (
from pandas.tseries import offsets
from pandas.tseries.offsets import Minute
def test_resample_origin_epoch_with_tz_day_vs_24h(unit):
    start, end = ('2000-10-01 23:30:00+0500', '2000-12-02 00:30:00+0500')
    rng = date_range(start, end, freq='7min').as_unit(unit)
    random_values = np.random.default_rng(2).standard_normal(len(rng))
    ts_1 = Series(random_values, index=rng)
    result_1 = ts_1.resample('D', origin='epoch').mean()
    result_2 = ts_1.resample('24h', origin='epoch').mean()
    tm.assert_series_equal(result_1, result_2)
    ts_no_tz = ts_1.tz_localize(None)
    result_3 = ts_no_tz.resample('D', origin='epoch').mean()
    result_4 = ts_no_tz.resample('24h', origin='epoch').mean()
    tm.assert_series_equal(result_1, result_3.tz_localize(rng.tz), check_freq=False)
    tm.assert_series_equal(result_1, result_4.tz_localize(rng.tz), check_freq=False)
    start, end = ('2000-10-01 23:30:00+0200', '2000-12-02 00:30:00+0200')
    rng = date_range(start, end, freq='7min').as_unit(unit)
    ts_2 = Series(random_values, index=rng)
    result_5 = ts_2.resample('D', origin='epoch').mean()
    result_6 = ts_2.resample('24h', origin='epoch').mean()
    tm.assert_series_equal(result_1.tz_localize(None), result_5.tz_localize(None))
    tm.assert_series_equal(result_1.tz_localize(None), result_6.tz_localize(None))