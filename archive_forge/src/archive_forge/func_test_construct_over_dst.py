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
def test_construct_over_dst(self, unit):
    pre_dst = Timestamp('2010-11-07 01:00:00').tz_localize('US/Pacific', ambiguous=True)
    pst_dst = Timestamp('2010-11-07 01:00:00').tz_localize('US/Pacific', ambiguous=False)
    expect_data = [Timestamp('2010-11-07 00:00:00', tz='US/Pacific'), pre_dst, pst_dst]
    expected = DatetimeIndex(expect_data, freq='h').as_unit(unit)
    result = date_range(start='2010-11-7', periods=3, freq='h', tz='US/Pacific', unit=unit)
    tm.assert_index_equal(result, expected)