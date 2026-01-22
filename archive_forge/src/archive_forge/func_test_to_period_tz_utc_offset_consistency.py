import dateutil.tz
from dateutil.tz import tzlocal
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import MONTHS
from pandas._libs.tslibs.offsets import MonthEnd
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', ['Etc/GMT-1', 'Etc/GMT+1'])
def test_to_period_tz_utc_offset_consistency(self, tz):
    ts = date_range('1/1/2000', '2/1/2000', tz='Etc/GMT-1')
    with tm.assert_produces_warning(UserWarning):
        result = ts.to_period()[0]
        expected = ts[0].to_period(ts.freq)
        assert result == expected