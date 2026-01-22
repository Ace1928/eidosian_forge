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
def test_groupby_with_dst_time_change(unit):
    index = DatetimeIndex([1478064900001000000, 1480037118776792000], tz='UTC').tz_convert('America/Chicago').as_unit(unit)
    df = DataFrame([1, 2], index=index)
    result = df.groupby(Grouper(freq='1d')).last()
    expected_index_values = date_range('2016-11-02', '2016-11-24', freq='d', tz='America/Chicago').as_unit(unit)
    index = DatetimeIndex(expected_index_values)
    expected = DataFrame([1.0] + [np.nan] * 21 + [2.0], index=index)
    tm.assert_frame_equal(result, expected)