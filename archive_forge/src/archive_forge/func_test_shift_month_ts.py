from datetime import datetime
import pytest
from pandas._libs.tslibs.ccalendar import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import roll_qtrday
from pandas import Timestamp
@pytest.mark.parametrize('months,day_opt,expected', [(1, 'start', Timestamp('1929-06-01')), (-3, 'end', Timestamp('1929-02-28')), (25, None, Timestamp('1931-06-5')), (-1, 31, Timestamp('1929-04-30'))])
def test_shift_month_ts(months, day_opt, expected):
    ts = Timestamp('1929-05-05')
    assert liboffsets.shift_month(ts, months, day_opt=day_opt) == expected