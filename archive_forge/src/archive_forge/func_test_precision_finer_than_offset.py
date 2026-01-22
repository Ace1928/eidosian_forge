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
def test_precision_finer_than_offset(self):
    result1 = date_range(start='2015-04-15 00:00:03', end='2016-04-22 00:00:00', freq='QE')
    result2 = date_range(start='2015-04-15 00:00:03', end='2015-06-22 00:00:04', freq='W')
    expected1_list = ['2015-06-30 00:00:03', '2015-09-30 00:00:03', '2015-12-31 00:00:03', '2016-03-31 00:00:03']
    expected2_list = ['2015-04-19 00:00:03', '2015-04-26 00:00:03', '2015-05-03 00:00:03', '2015-05-10 00:00:03', '2015-05-17 00:00:03', '2015-05-24 00:00:03', '2015-05-31 00:00:03', '2015-06-07 00:00:03', '2015-06-14 00:00:03', '2015-06-21 00:00:03']
    expected1 = DatetimeIndex(expected1_list, dtype='datetime64[ns]', freq='QE-DEC', tz=None)
    expected2 = DatetimeIndex(expected2_list, dtype='datetime64[ns]', freq='W-SUN', tz=None)
    tm.assert_index_equal(result1, expected1)
    tm.assert_index_equal(result2, expected2)