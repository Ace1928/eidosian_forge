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
@pytest.mark.parametrize('freq', ['B', 'C'])
def test_resample_c_b_closed_right(freq: str, unit):
    dti = date_range(start='2020-01-31', freq='1min', periods=6000, unit=unit)
    df = DataFrame({'ts': dti}, index=dti)
    grouped = df.resample(freq, closed='right')
    result = grouped.last()
    exp_dti = DatetimeIndex([datetime(2020, 1, 30), datetime(2020, 1, 31), datetime(2020, 2, 3), datetime(2020, 2, 4)], freq=freq).as_unit(unit)
    expected = DataFrame({'ts': [datetime(2020, 1, 31), datetime(2020, 2, 3), datetime(2020, 2, 4), datetime(2020, 2, 4, 3, 59)]}, index=exp_dti).astype(f'M8[{unit}]')
    tm.assert_frame_equal(result, expected)