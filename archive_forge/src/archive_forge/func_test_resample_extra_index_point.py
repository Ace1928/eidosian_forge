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
def test_resample_extra_index_point(unit):
    index = date_range(start='20150101', end='20150331', freq='BME').as_unit(unit)
    expected = DataFrame({'A': Series([21, 41, 63], index=index)})
    index = date_range(start='20150101', end='20150331', freq='B').as_unit(unit)
    df = DataFrame({'A': Series(range(len(index)), index=index)}, dtype='int64')
    result = df.resample('BME').last()
    tm.assert_frame_equal(result, expected)