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
def test_date_range_with_fixed_tz(self):
    off = FixedOffset(420, '+07:00')
    start = datetime(2012, 3, 11, 5, 0, 0, tzinfo=off)
    end = datetime(2012, 6, 11, 5, 0, 0, tzinfo=off)
    rng = date_range(start=start, end=end)
    assert off == rng.tz
    rng2 = date_range(start, periods=len(rng), tz=off)
    tm.assert_index_equal(rng, rng2)
    rng3 = date_range('3/11/2012 05:00:00+07:00', '6/11/2012 05:00:00+07:00')
    assert (rng.values == rng3.values).all()