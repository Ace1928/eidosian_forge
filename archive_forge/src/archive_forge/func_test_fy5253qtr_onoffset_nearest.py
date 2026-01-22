from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytest
from pandas import Timestamp
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
def test_fy5253qtr_onoffset_nearest():
    ts = Timestamp('1985-09-02 23:57:46.232550356-0300', tz='Atlantic/Bermuda')
    offset = FY5253Quarter(n=3, qtr_with_extra_week=1, startingMonth=2, variation='nearest', weekday=0)
    fast = offset.is_on_offset(ts)
    slow = ts + offset - offset == ts
    assert fast == slow