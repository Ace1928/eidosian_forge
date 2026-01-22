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
@pytest.mark.parametrize('freq, expected_kwargs', [['YE-DEC', {'start': '1990', 'end': '2000', 'freq': 'Y-DEC'}], ['YE-JUN', {'start': '1990', 'end': '2000', 'freq': 'Y-JUN'}], ['ME', {'start': '1990-01', 'end': '2000-01', 'freq': 'M'}]])
def test_resample_timestamp_to_period(simple_date_range_series, freq, expected_kwargs, unit):
    ts = simple_date_range_series('1/1/1990', '1/1/2000')
    ts.index = ts.index.as_unit(unit)
    msg = "The 'kind' keyword in Series.resample is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = ts.resample(freq, kind='period').mean()
    expected = ts.resample(freq).mean()
    expected.index = period_range(**expected_kwargs)
    tm.assert_series_equal(result, expected)