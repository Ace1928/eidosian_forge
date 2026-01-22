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
def test_resample_to_quarterly(self, simple_period_range_series, month):
    ts = simple_period_range_series('1990', '1992', freq=f'Y-{month}')
    quar_ts = ts.resample(f'Q-{month}').ffill()
    stamps = ts.to_timestamp('D', how='start')
    qdates = period_range(ts.index[0].asfreq('D', 'start'), ts.index[-1].asfreq('D', 'end'), freq=f'Q-{month}')
    expected = stamps.reindex(qdates.to_timestamp('D', 's'), method='ffill')
    expected.index = qdates
    tm.assert_series_equal(quar_ts, expected)