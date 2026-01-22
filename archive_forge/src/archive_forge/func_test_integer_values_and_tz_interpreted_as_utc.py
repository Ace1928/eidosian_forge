from __future__ import annotations
from datetime import (
from functools import partial
from operator import attrgetter
import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_integer_values_and_tz_interpreted_as_utc(self):
    val = np.datetime64('2000-01-01 00:00:00', 'ns')
    values = np.array([val.view('i8')])
    result = DatetimeIndex(values).tz_localize('US/Central')
    expected = DatetimeIndex(['2000-01-01T00:00:00'], dtype='M8[ns, US/Central]')
    tm.assert_index_equal(result, expected)
    with tm.assert_produces_warning(None):
        result = DatetimeIndex(values, tz='UTC')
    expected = DatetimeIndex(['2000-01-01T00:00:00'], dtype='M8[ns, UTC]')
    tm.assert_index_equal(result, expected)