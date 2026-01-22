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
def test_range_tz_pytz(self):
    tz = timezone('US/Eastern')
    start = tz.localize(datetime(2011, 1, 1))
    end = tz.localize(datetime(2011, 1, 3))
    dr = date_range(start=start, periods=3)
    assert dr.tz.zone == tz.zone
    assert dr[0] == start
    assert dr[2] == end
    dr = date_range(end=end, periods=3)
    assert dr.tz.zone == tz.zone
    assert dr[0] == start
    assert dr[2] == end
    dr = date_range(start=start, end=end)
    assert dr.tz.zone == tz.zone
    assert dr[0] == start
    assert dr[2] == end