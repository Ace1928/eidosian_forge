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
def test_date_range_convenience_periods(self, unit):
    result = date_range('2018-04-24', '2018-04-27', periods=3, unit=unit)
    expected = DatetimeIndex(['2018-04-24 00:00:00', '2018-04-25 12:00:00', '2018-04-27 00:00:00'], dtype=f'M8[{unit}]', freq=None)
    tm.assert_index_equal(result, expected)
    result = date_range('2018-04-01 01:00:00', '2018-04-01 04:00:00', tz='Australia/Sydney', periods=3, unit=unit)
    expected = DatetimeIndex([Timestamp('2018-04-01 01:00:00+1100', tz='Australia/Sydney'), Timestamp('2018-04-01 02:00:00+1000', tz='Australia/Sydney'), Timestamp('2018-04-01 04:00:00+1000', tz='Australia/Sydney')]).as_unit(unit)
    tm.assert_index_equal(result, expected)