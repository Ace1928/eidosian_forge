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
def test_date_range_normalize(self):
    snap = datetime.today()
    n = 50
    rng = date_range(snap, periods=n, normalize=False, freq='2D')
    offset = timedelta(2)
    expected = DatetimeIndex([snap + i * offset for i in range(n)], dtype='M8[ns]', freq=offset)
    tm.assert_index_equal(rng, expected)
    rng = date_range('1/1/2000 08:15', periods=n, normalize=False, freq='B')
    the_time = time(8, 15)
    for val in rng:
        assert val.time() == the_time