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
@pytest.mark.parametrize('month', MONTHS)
@pytest.mark.parametrize('convention', ['start', 'end'])
@pytest.mark.parametrize(('offset', 'period'), [('D', 'D'), ('B', 'B'), ('ME', 'M')])
def test_quarterly_upsample(self, month, offset, period, convention, simple_period_range_series):
    freq = f'Q-{month}'
    ts = simple_period_range_series('1/1/1990', '12/31/1995', freq=freq)
    warn = FutureWarning if period == 'B' else None
    msg = 'PeriodDtype\\[B\\] is deprecated'
    if warn is None:
        msg = 'Resampling with a PeriodIndex is deprecated'
        warn = FutureWarning
    with tm.assert_produces_warning(warn, match=msg):
        result = ts.resample(period, convention=convention).ffill()
        expected = result.to_timestamp(period, how=convention)
        expected = expected.asfreq(offset, 'ffill').to_period()
    tm.assert_series_equal(result, expected)