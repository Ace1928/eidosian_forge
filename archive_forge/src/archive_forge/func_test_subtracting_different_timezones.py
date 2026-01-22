from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas._testing as tm
def test_subtracting_different_timezones(self, tz_aware_fixture):
    t_raw = Timestamp('20130101')
    t_UTC = t_raw.tz_localize('UTC')
    t_diff = t_UTC.tz_convert(tz_aware_fixture) + Timedelta('0 days 05:00:00')
    result = t_diff - t_UTC
    assert isinstance(result, Timedelta)
    assert result == Timedelta('0 days 05:00:00')