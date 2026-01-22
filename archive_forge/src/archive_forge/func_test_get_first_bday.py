from datetime import datetime
import pytest
from pandas._libs.tslibs.ccalendar import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import roll_qtrday
from pandas import Timestamp
@pytest.mark.parametrize('dt,exp_week_day,exp_first_day', [(datetime(2017, 4, 1), 5, 3), (datetime(1993, 10, 1), 4, 1)])
def test_get_first_bday(dt, exp_week_day, exp_first_day):
    assert dt.weekday() == exp_week_day
    assert get_firstbday(dt.year, dt.month) == exp_first_day