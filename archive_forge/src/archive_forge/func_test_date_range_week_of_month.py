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
def test_date_range_week_of_month(self, unit):
    result = date_range(start='20110101', periods=1, freq='WOM-1MON', unit=unit)
    expected = DatetimeIndex(['2011-01-03'], dtype=f'M8[{unit}]', freq='WOM-1MON')
    tm.assert_index_equal(result, expected)
    result2 = date_range(start='20110101', periods=2, freq='WOM-1MON', unit=unit)
    expected2 = DatetimeIndex(['2011-01-03', '2011-02-07'], dtype=f'M8[{unit}]', freq='WOM-1MON')
    tm.assert_index_equal(result2, expected2)