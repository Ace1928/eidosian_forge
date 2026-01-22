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
@pytest.mark.parametrize('freq', ['1D', '10h', '5Min', '10s'])
@pytest.mark.parametrize('rule', ['YE', '3ME', '15D', '30h', '15Min', '30s'])
def test_nearest_upsample_with_limit(tz_aware_fixture, freq, rule, unit):
    rng = date_range('1/1/2000', periods=3, freq=freq, tz=tz_aware_fixture).as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
    result = ts.resample(rule).nearest(limit=2)
    expected = ts.reindex(result.index, method='nearest', limit=2)
    tm.assert_series_equal(result, expected)