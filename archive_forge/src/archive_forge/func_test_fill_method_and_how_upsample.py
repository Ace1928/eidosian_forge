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
def test_fill_method_and_how_upsample(self):
    s = Series(np.arange(9, dtype='int64'), index=date_range('2010-01-01', periods=9, freq='QE'))
    last = s.resample('ME').ffill()
    both = s.resample('ME').ffill().resample('ME').last().astype('int64')
    tm.assert_series_equal(last, both)