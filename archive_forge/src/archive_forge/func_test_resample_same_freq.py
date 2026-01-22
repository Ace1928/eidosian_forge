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
def test_resample_same_freq(self, resample_method):
    series = Series(range(3), index=period_range(start='2000', periods=3, freq='M'))
    expected = series
    result = getattr(series.resample('M'), resample_method)()
    tm.assert_series_equal(result, expected)