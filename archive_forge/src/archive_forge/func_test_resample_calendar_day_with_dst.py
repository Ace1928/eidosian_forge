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
@pytest.mark.parametrize('first,last,freq_in,freq_out,exp_last', [('2020-03-28', '2020-03-31', 'D', '24h', '2020-03-30 01:00'), ('2020-03-28', '2020-10-27', 'D', '24h', '2020-10-27 00:00'), ('2020-10-25', '2020-10-27', 'D', '24h', '2020-10-26 23:00'), ('2020-03-28', '2020-03-31', '24h', 'D', '2020-03-30 00:00'), ('2020-03-28', '2020-10-27', '24h', 'D', '2020-10-27 00:00'), ('2020-10-25', '2020-10-27', '24h', 'D', '2020-10-26 00:00')])
def test_resample_calendar_day_with_dst(first: str, last: str, freq_in: str, freq_out: str, exp_last: str, unit):
    ts = Series(1.0, date_range(first, last, freq=freq_in, tz='Europe/Amsterdam').as_unit(unit))
    result = ts.resample(freq_out).ffill()
    expected = Series(1.0, date_range(first, exp_last, freq=freq_out, tz='Europe/Amsterdam').as_unit(unit))
    tm.assert_series_equal(result, expected)