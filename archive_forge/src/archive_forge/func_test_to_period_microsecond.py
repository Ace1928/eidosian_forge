import dateutil.tz
from dateutil.tz import tzlocal
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import MONTHS
from pandas._libs.tslibs.offsets import MonthEnd
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
def test_to_period_microsecond(self):
    index = DatetimeIndex([Timestamp('2007-01-01 10:11:12.123456Z'), Timestamp('2007-01-01 10:11:13.789123Z')])
    with tm.assert_produces_warning(UserWarning):
        period = index.to_period(freq='us')
    assert 2 == len(period)
    assert period[0] == Period('2007-01-01 10:11:12.123456Z', 'us')
    assert period[1] == Period('2007-01-01 10:11:13.789123Z', 'us')