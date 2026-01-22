from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
from pandas.tests.tseries.offsets.common import assert_offset_equal
from pandas.tseries.holiday import USFederalHolidayCalendar
@pytest.mark.parametrize('weekmask, expected_time, mult', [['Mon Tue Wed Thu Fri Sat', '2018-11-10 09:00:00', 10], ['Tue Wed Thu Fri Sat', '2018-11-13 08:00:00', 18]])
def test_custom_businesshour_weekmask_and_holidays(weekmask, expected_time, mult):
    holidays = ['2018-11-09']
    bh = CustomBusinessHour(start='08:00', end='17:00', weekmask=weekmask, holidays=holidays)
    result = Timestamp('2018-11-08 08:00') + mult * bh
    expected = Timestamp(expected_time)
    assert result == expected