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
def test_date_range_business_hour2(self, unit):
    idx1 = date_range(start='2014-07-04 15:00', end='2014-07-08 10:00', freq='bh', unit=unit)
    idx2 = date_range(start='2014-07-04 15:00', periods=12, freq='bh', unit=unit)
    idx3 = date_range(end='2014-07-08 10:00', periods=12, freq='bh', unit=unit)
    expected = DatetimeIndex(['2014-07-04 15:00', '2014-07-04 16:00', '2014-07-07 09:00', '2014-07-07 10:00', '2014-07-07 11:00', '2014-07-07 12:00', '2014-07-07 13:00', '2014-07-07 14:00', '2014-07-07 15:00', '2014-07-07 16:00', '2014-07-08 09:00', '2014-07-08 10:00'], dtype=f'M8[{unit}]', freq='bh')
    tm.assert_index_equal(idx1, expected)
    tm.assert_index_equal(idx2, expected)
    tm.assert_index_equal(idx3, expected)
    idx4 = date_range(start='2014-07-04 15:45', end='2014-07-08 10:45', freq='bh', unit=unit)
    idx5 = date_range(start='2014-07-04 15:45', periods=12, freq='bh', unit=unit)
    idx6 = date_range(end='2014-07-08 10:45', periods=12, freq='bh', unit=unit)
    expected2 = expected + Timedelta(minutes=45).as_unit(unit)
    expected2.freq = 'bh'
    tm.assert_index_equal(idx4, expected2)
    tm.assert_index_equal(idx5, expected2)
    tm.assert_index_equal(idx6, expected2)