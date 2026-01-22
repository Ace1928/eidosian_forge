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
@pytest.mark.parametrize('freq', ['min', '5min', '15min', '30min', '4h', '12h'])
def test_resample_anchored_ticks(freq, unit):
    rng = date_range('1/1/2000 04:00:00', periods=86400, freq='s').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    ts[:2] = np.nan
    result = ts[2:].resample(freq, closed='left', label='left').mean()
    expected = ts.resample(freq, closed='left', label='left').mean()
    tm.assert_series_equal(result, expected)