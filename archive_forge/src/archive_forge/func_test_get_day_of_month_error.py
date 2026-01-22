from datetime import datetime
import pytest
from pandas._libs.tslibs.ccalendar import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import roll_qtrday
from pandas import Timestamp
def test_get_day_of_month_error():
    dt = datetime(2017, 11, 15)
    day_opt = 'foo'
    with pytest.raises(ValueError, match=day_opt):
        roll_qtrday(dt, n=3, month=11, day_opt=day_opt, modby=12)