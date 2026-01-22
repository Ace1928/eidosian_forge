from datetime import timezone
import pytest
import pytz
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
def test_series_tz_localize_ambiguous_bool(self):
    ts = Timestamp('2015-11-01 01:00:03')
    expected0 = Timestamp('2015-11-01 01:00:03-0500', tz='US/Central')
    expected1 = Timestamp('2015-11-01 01:00:03-0600', tz='US/Central')
    ser = Series([ts])
    expected0 = Series([expected0])
    expected1 = Series([expected1])
    with tm.external_error_raised(pytz.AmbiguousTimeError):
        ser.dt.tz_localize('US/Central')
    result = ser.dt.tz_localize('US/Central', ambiguous=True)
    tm.assert_series_equal(result, expected0)
    result = ser.dt.tz_localize('US/Central', ambiguous=[True])
    tm.assert_series_equal(result, expected0)
    result = ser.dt.tz_localize('US/Central', ambiguous=False)
    tm.assert_series_equal(result, expected1)
    result = ser.dt.tz_localize('US/Central', ambiguous=[False])
    tm.assert_series_equal(result, expected1)