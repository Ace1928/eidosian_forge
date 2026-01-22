import dateutil.tz
from dateutil.tz import tzlocal
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import MONTHS
from pandas._libs.tslibs.offsets import MonthEnd
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
def test_to_period_infer(self):
    rng = date_range(start='2019-12-22 06:40:00+00:00', end='2019-12-22 08:45:00+00:00', freq='5min')
    with tm.assert_produces_warning(UserWarning):
        pi1 = rng.to_period('5min')
    with tm.assert_produces_warning(UserWarning):
        pi2 = rng.to_period()
    tm.assert_index_equal(pi1, pi2)