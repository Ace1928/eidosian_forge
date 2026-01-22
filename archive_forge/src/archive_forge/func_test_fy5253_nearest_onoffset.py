from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytest
from pandas import Timestamp
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
def test_fy5253_nearest_onoffset():
    offset = FY5253(n=3, startingMonth=7, variation='nearest', weekday=2)
    ts = Timestamp('2032-07-28 00:12:59.035729419+0000', tz='Africa/Dakar')
    fast = offset.is_on_offset(ts)
    slow = ts + offset - offset == ts
    assert fast == slow