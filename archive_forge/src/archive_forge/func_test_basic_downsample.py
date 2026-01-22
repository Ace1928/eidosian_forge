from datetime import datetime
import warnings
import dateutil
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import _get_period_range_edges
from pandas.tseries import offsets
def test_basic_downsample(self, simple_period_range_series):
    ts = simple_period_range_series('1/1/1990', '6/30/1995', freq='M')
    result = ts.resample('Y-DEC').mean()
    expected = ts.groupby(ts.index.year).mean()
    expected.index = period_range('1/1/1990', '6/30/1995', freq='Y-DEC')
    tm.assert_series_equal(result, expected)
    tm.assert_series_equal(ts.resample('Y-DEC').mean(), result)
    tm.assert_series_equal(ts.resample('Y').mean(), result)