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
@pytest.mark.parametrize('freq', ['5min'])
@pytest.mark.parametrize('kind', ['period', None, 'timestamp'])
def test_resample_5minute(self, freq, kind):
    rng = period_range('1/1/2000', '1/5/2000', freq='min')
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    expected = ts.to_timestamp().resample(freq).mean()
    if kind != 'timestamp':
        expected = expected.to_period(freq)
    msg = "The 'kind' keyword in Series.resample is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = ts.resample(freq, kind=kind).mean()
    tm.assert_series_equal(result, expected)