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
@pytest.mark.parametrize('dtype', ['int64', 'int32', 'float64', pytest.param('float32', marks=pytest.mark.xfail(reason='Empty groups cause x.mean() to return float64'))])
def test_resample_median_bug_1688(dtype, unit):
    dti = DatetimeIndex([datetime(2012, 1, 1, 0, 0, 0), datetime(2012, 1, 1, 0, 5, 0)]).as_unit(unit)
    df = DataFrame([1, 2], index=dti, dtype=dtype)
    result = df.resample('min').apply(lambda x: x.mean())
    exp = df.asfreq('min')
    tm.assert_frame_equal(result, exp)
    result = df.resample('min').median()
    exp = df.asfreq('min')
    tm.assert_frame_equal(result, exp)