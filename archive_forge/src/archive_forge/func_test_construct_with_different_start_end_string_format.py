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
def test_construct_with_different_start_end_string_format(self, unit):
    result = date_range('2013-01-01 00:00:00+09:00', '2013/01/01 02:00:00+09:00', freq='h', unit=unit)
    expected = DatetimeIndex([Timestamp('2013-01-01 00:00:00+09:00'), Timestamp('2013-01-01 01:00:00+09:00'), Timestamp('2013-01-01 02:00:00+09:00')], freq='h').as_unit(unit)
    tm.assert_index_equal(result, expected)