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
def test_resample_apply_with_additional_args(series, unit):

    def f(data, add_arg):
        return np.mean(data) * add_arg
    series.index = series.index.as_unit(unit)
    multiplier = 10
    result = series.resample('D').apply(f, multiplier)
    expected = series.resample('D').mean().multiply(multiplier)
    tm.assert_series_equal(result, expected)
    result = series.resample('D').apply(f, add_arg=multiplier)
    expected = series.resample('D').mean().multiply(multiplier)
    tm.assert_series_equal(result, expected)