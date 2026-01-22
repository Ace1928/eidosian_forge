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
def test_date_range_non_nano(self):
    start = np.datetime64('1066-10-14')
    end = np.datetime64('2305-07-13')
    dti = date_range(start, end, freq='D', unit='s')
    assert dti.freq == 'D'
    assert dti.dtype == 'M8[s]'
    exp = np.arange(start.astype('M8[s]').view('i8'), (end + 1).astype('M8[s]').view('i8'), 24 * 3600).view('M8[s]')
    tm.assert_numpy_array_equal(dti.to_numpy(), exp)