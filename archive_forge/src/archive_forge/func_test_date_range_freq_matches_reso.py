from datetime import (
import re
import numpy as np
import pytest
import pytz
from pytz import timezone
from pandas._libs.tslibs import timezones
from pandas._libs.tslibs.offsets import (
from pandas.errors import OutOfBoundsDatetime
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.datetimes import _generate_range as generate_range
from pandas.tests.indexes.datetimes.test_timezones import (
from pandas.tseries.holiday import USFederalHolidayCalendar
def test_date_range_freq_matches_reso(self):
    dti = date_range('2016-01-01', '2016-01-01 00:00:01', freq='ms', unit='ms')
    rng = np.arange(1451606400000, 1451606401001, dtype=np.int64)
    expected = DatetimeIndex(rng.view('M8[ms]'), freq='ms')
    tm.assert_index_equal(dti, expected)
    dti = date_range('2016-01-01', '2016-01-01 00:00:01', freq='us', unit='us')
    rng = np.arange(1451606400000000, 1451606401000001, dtype=np.int64)
    expected = DatetimeIndex(rng.view('M8[us]'), freq='us')
    tm.assert_index_equal(dti, expected)
    dti = date_range('2016-01-01', '2016-01-01 00:00:00.001', freq='ns', unit='ns')
    rng = np.arange(1451606400000000000, 1451606400001000001, dtype=np.int64)
    expected = DatetimeIndex(rng.view('M8[ns]'), freq='ns')
    tm.assert_index_equal(dti, expected)