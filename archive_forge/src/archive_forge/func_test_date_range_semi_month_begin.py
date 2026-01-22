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
def test_date_range_semi_month_begin(self, unit):
    dates = [datetime(2007, 12, 15), datetime(2008, 1, 1), datetime(2008, 1, 15), datetime(2008, 2, 1), datetime(2008, 2, 15), datetime(2008, 3, 1), datetime(2008, 3, 15), datetime(2008, 4, 1), datetime(2008, 4, 15), datetime(2008, 5, 1), datetime(2008, 5, 15), datetime(2008, 6, 1), datetime(2008, 6, 15), datetime(2008, 7, 1), datetime(2008, 7, 15), datetime(2008, 8, 1), datetime(2008, 8, 15), datetime(2008, 9, 1), datetime(2008, 9, 15), datetime(2008, 10, 1), datetime(2008, 10, 15), datetime(2008, 11, 1), datetime(2008, 11, 15), datetime(2008, 12, 1), datetime(2008, 12, 15)]
    result = date_range(start=dates[0], end=dates[-1], freq='SMS', unit=unit)
    exp = DatetimeIndex(dates, dtype=f'M8[{unit}]', freq='SMS')
    tm.assert_index_equal(result, exp)