import dateutil.tz
from dateutil.tz import tzlocal
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import MONTHS
from pandas._libs.tslibs.offsets import MonthEnd
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('off', ['BYE', 'YS', 'BYS'])
def test_to_period_annualish(self, off):
    rng = date_range('01-Jan-2012', periods=8, freq=off)
    prng = rng.to_period()
    assert prng.freq == 'YE-DEC'