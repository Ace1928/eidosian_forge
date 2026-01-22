from datetime import datetime
import pytest
from pytz import utc
from pandas import (
import pandas._testing as tm
from pandas.tseries.holiday import (
def test_half_open_interval_with_observance():
    holiday_1 = Holiday('Arbitrary Holiday - start 2022-03-14', start_date=datetime(2022, 3, 14), month=3, day=14, observance=next_monday)
    holiday_2 = Holiday('Arbitrary Holiday 2 - end 2022-03-20', end_date=datetime(2022, 3, 20), month=3, day=20, observance=next_monday)

    class TestHolidayCalendar(AbstractHolidayCalendar):
        rules = [USMartinLutherKingJr, holiday_1, holiday_2, USLaborDay]
    start = Timestamp('2022-08-01')
    end = Timestamp('2022-08-31')
    year_offset = DateOffset(years=5)
    expected_results = DatetimeIndex([], dtype='datetime64[ns]', freq=None)
    test_cal = TestHolidayCalendar()
    date_interval_low = test_cal.holidays(start - year_offset, end - year_offset)
    date_window_edge = test_cal.holidays(start, end)
    date_interval_high = test_cal.holidays(start + year_offset, end + year_offset)
    tm.assert_index_equal(date_interval_low, expected_results)
    tm.assert_index_equal(date_window_edge, expected_results)
    tm.assert_index_equal(date_interval_high, expected_results)