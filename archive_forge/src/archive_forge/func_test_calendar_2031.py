from datetime import datetime
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries.holiday import (
def test_calendar_2031():

    class testCalendar(AbstractHolidayCalendar):
        rules = [USLaborDay]
    cal = testCalendar()
    workDay = offsets.CustomBusinessDay(calendar=cal)
    Sat_before_Labor_Day_2031 = to_datetime('2031-08-30')
    next_working_day = Sat_before_Labor_Day_2031 + 0 * workDay
    assert next_working_day == to_datetime('2031-09-02')