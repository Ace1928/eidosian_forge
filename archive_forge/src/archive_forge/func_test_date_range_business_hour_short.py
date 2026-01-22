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
def test_date_range_business_hour_short(self, unit):
    idx4 = date_range(start='2014-07-01 10:00', freq='bh', periods=1, unit=unit)
    expected4 = DatetimeIndex(['2014-07-01 10:00'], dtype=f'M8[{unit}]', freq='bh')
    tm.assert_index_equal(idx4, expected4)