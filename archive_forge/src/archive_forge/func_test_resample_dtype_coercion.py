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
def test_resample_dtype_coercion(unit):
    pytest.importorskip('scipy.interpolate')
    df = {'a': [1, 3, 1, 4]}
    df = DataFrame(df, index=date_range('2017-01-01', '2017-01-04').as_unit(unit))
    expected = df.astype('float64').resample('h').mean()['a'].interpolate('cubic')
    result = df.resample('h')['a'].mean().interpolate('cubic')
    tm.assert_series_equal(result, expected)
    result = df.resample('h').mean()['a'].interpolate('cubic')
    tm.assert_series_equal(result, expected)