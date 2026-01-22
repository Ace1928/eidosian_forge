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
@td.skip_if_no('pyarrow')
@pytest.mark.parametrize('tz', [None, pytest.param('UTC', marks=pytest.mark.xfail(condition=is_platform_windows(), reason='TODO: Set ARROW_TIMEZONE_DATABASE env var in CI'))])
def test_arrow_timestamp_resample(tz):
    idx = Series(date_range('2020-01-01', periods=5), dtype='timestamp[ns][pyarrow]')
    if tz is not None:
        idx = idx.dt.tz_localize(tz)
    expected = Series(np.arange(5, dtype=np.float64), index=idx)
    result = expected.resample('1D').mean()
    tm.assert_series_equal(result, expected)