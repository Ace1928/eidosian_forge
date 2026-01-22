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
def test_date_range_businesshour(self, unit):
    idx = DatetimeIndex(['2014-07-04 09:00', '2014-07-04 10:00', '2014-07-04 11:00', '2014-07-04 12:00', '2014-07-04 13:00', '2014-07-04 14:00', '2014-07-04 15:00', '2014-07-04 16:00'], dtype=f'M8[{unit}]', freq='bh')
    rng = date_range('2014-07-04 09:00', '2014-07-04 16:00', freq='bh', unit=unit)
    tm.assert_index_equal(idx, rng)
    idx = DatetimeIndex(['2014-07-04 16:00', '2014-07-07 09:00'], dtype=f'M8[{unit}]', freq='bh')
    rng = date_range('2014-07-04 16:00', '2014-07-07 09:00', freq='bh', unit=unit)
    tm.assert_index_equal(idx, rng)
    idx = DatetimeIndex(['2014-07-04 09:00', '2014-07-04 10:00', '2014-07-04 11:00', '2014-07-04 12:00', '2014-07-04 13:00', '2014-07-04 14:00', '2014-07-04 15:00', '2014-07-04 16:00', '2014-07-07 09:00', '2014-07-07 10:00', '2014-07-07 11:00', '2014-07-07 12:00', '2014-07-07 13:00', '2014-07-07 14:00', '2014-07-07 15:00', '2014-07-07 16:00', '2014-07-08 09:00', '2014-07-08 10:00', '2014-07-08 11:00', '2014-07-08 12:00', '2014-07-08 13:00', '2014-07-08 14:00', '2014-07-08 15:00', '2014-07-08 16:00'], dtype=f'M8[{unit}]', freq='bh')
    rng = date_range('2014-07-04 09:00', '2014-07-08 16:00', freq='bh', unit=unit)
    tm.assert_index_equal(idx, rng)