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
def test_resample_integerarray(unit):
    ts = Series(range(9), index=date_range('1/1/2000', periods=9, freq='min').as_unit(unit), dtype='Int64')
    result = ts.resample('3min').sum()
    expected = Series([3, 12, 21], index=date_range('1/1/2000', periods=3, freq='3min').as_unit(unit), dtype='Int64')
    tm.assert_series_equal(result, expected)
    result = ts.resample('3min').mean()
    expected = Series([1, 4, 7], index=date_range('1/1/2000', periods=3, freq='3min').as_unit(unit), dtype='Float64')
    tm.assert_series_equal(result, expected)