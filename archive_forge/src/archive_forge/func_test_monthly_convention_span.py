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
@pytest.mark.xfail(reason='Commented out for more than 3 years. Should this work?')
def test_monthly_convention_span(self):
    rng = period_range('2000-01', periods=3, freq='ME')
    ts = Series(np.arange(3), index=rng)
    exp_index = period_range('2000-01-01', '2000-03-31', freq='D')
    expected = ts.asfreq('D', how='end').reindex(exp_index)
    expected = expected.fillna(method='bfill')
    result = ts.resample('D').mean()
    tm.assert_series_equal(result, expected)