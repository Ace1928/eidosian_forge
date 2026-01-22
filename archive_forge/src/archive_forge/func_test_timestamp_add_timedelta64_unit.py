from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas._testing as tm
@pytest.mark.parametrize('other,expected_difference', [(np.timedelta64(-123, 'ns'), -123), (np.timedelta64(1234567898, 'ns'), 1234567898), (np.timedelta64(-123, 'us'), -123000), (np.timedelta64(-123, 'ms'), -123000000)])
def test_timestamp_add_timedelta64_unit(self, other, expected_difference):
    now = datetime.now(timezone.utc)
    ts = Timestamp(now).as_unit('ns')
    result = ts + other
    valdiff = result._value - ts._value
    assert valdiff == expected_difference
    ts2 = Timestamp(now)
    assert ts2 + other == result