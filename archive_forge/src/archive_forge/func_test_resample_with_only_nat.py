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
def test_resample_with_only_nat(self):
    pi = PeriodIndex([pd.NaT] * 3, freq='s')
    frame = DataFrame([2, 3, 5], index=pi, columns=['a'])
    expected_index = PeriodIndex(data=[], freq=pi.freq)
    expected = DataFrame(index=expected_index, columns=['a'], dtype='float64')
    result = frame.resample('1s').mean()
    tm.assert_frame_equal(result, expected)