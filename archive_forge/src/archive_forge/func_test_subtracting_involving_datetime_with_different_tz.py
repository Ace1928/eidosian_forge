from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas._testing as tm
def test_subtracting_involving_datetime_with_different_tz(self):
    t1 = datetime(2013, 1, 1, tzinfo=timezone(timedelta(hours=-5)))
    t2 = Timestamp('20130101').tz_localize('CET')
    result = t1 - t2
    assert isinstance(result, Timedelta)
    assert result == Timedelta('0 days 06:00:00')
    result = t2 - t1
    assert isinstance(result, Timedelta)
    assert result == Timedelta('-1 days +18:00:00')