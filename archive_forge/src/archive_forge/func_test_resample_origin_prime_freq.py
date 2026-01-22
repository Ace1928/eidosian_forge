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
def test_resample_origin_prime_freq(unit):
    start, end = ('2000-10-01 23:30:00', '2000-10-02 00:30:00')
    rng = date_range(start, end, freq='7min').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    exp_rng = date_range('2000-10-01 23:14:00', '2000-10-02 00:22:00', freq='17min').as_unit(unit)
    resampled = ts.resample('17min').mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    resampled = ts.resample('17min', origin='start_day').mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    exp_rng = date_range('2000-10-01 23:30:00', '2000-10-02 00:21:00', freq='17min').as_unit(unit)
    resampled = ts.resample('17min', origin='start').mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    resampled = ts.resample('17min', offset='23h30min').mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    resampled = ts.resample('17min', origin='start_day', offset='23h30min').mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    exp_rng = date_range('2000-10-01 23:18:00', '2000-10-02 00:26:00', freq='17min').as_unit(unit)
    resampled = ts.resample('17min', origin='epoch').mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    exp_rng = date_range('2000-10-01 23:24:00', '2000-10-02 00:15:00', freq='17min').as_unit(unit)
    resampled = ts.resample('17min', origin='2000-01-01').mean()
    tm.assert_index_equal(resampled.index, exp_rng)