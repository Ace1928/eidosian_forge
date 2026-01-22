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
@pytest.mark.parametrize('freq,expected_vals', [('M', [31, 29, 31, 9]), ('2M', [31 + 29, 31 + 9])])
def test_resample_count(self, freq, expected_vals):
    series = Series(1, index=period_range(start='2000', periods=100))
    result = series.resample(freq).count()
    expected_index = period_range(start='2000', freq=freq, periods=len(expected_vals))
    expected = Series(expected_vals, index=expected_index)
    tm.assert_series_equal(result, expected)