import dateutil.tz
from dateutil.tz import tzlocal
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import MONTHS
from pandas._libs.tslibs.offsets import MonthEnd
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('freq', ['2BMS', '1SME-15'])
def test_to_period_offsets_not_supported(self, freq):
    msg = f'{freq[1:]} is not supported as period frequency'
    ts = date_range('1/1/2012', periods=4, freq=freq)
    with pytest.raises(ValueError, match=msg):
        ts.to_period()