from datetime import datetime
import pytest
from pandas._libs.tslibs.ccalendar import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import roll_qtrday
from pandas import Timestamp
@pytest.mark.parametrize('months,day_opt,expected', [(0, 15, datetime(2017, 11, 15)), (0, None, datetime(2017, 11, 30)), (1, 'start', datetime(2017, 12, 1)), (-145, 'end', datetime(2005, 10, 31)), (0, 'business_end', datetime(2017, 11, 30)), (0, 'business_start', datetime(2017, 11, 1))])
def test_shift_month_dt(months, day_opt, expected):
    dt = datetime(2017, 11, 30)
    assert liboffsets.shift_month(dt, months, day_opt=day_opt) == expected