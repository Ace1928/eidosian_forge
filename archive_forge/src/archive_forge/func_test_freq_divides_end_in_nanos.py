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
def test_freq_divides_end_in_nanos(self):
    result_1 = date_range('2005-01-12 10:00', '2005-01-12 16:00', freq='345min')
    result_2 = date_range('2005-01-13 10:00', '2005-01-13 16:00', freq='345min')
    expected_1 = DatetimeIndex(['2005-01-12 10:00:00', '2005-01-12 15:45:00'], dtype='datetime64[ns]', freq='345min', tz=None)
    expected_2 = DatetimeIndex(['2005-01-13 10:00:00', '2005-01-13 15:45:00'], dtype='datetime64[ns]', freq='345min', tz=None)
    tm.assert_index_equal(result_1, expected_1)
    tm.assert_index_equal(result_2, expected_2)