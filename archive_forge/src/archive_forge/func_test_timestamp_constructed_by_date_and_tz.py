import calendar
from datetime import (
import zoneinfo
import dateutil.tz
from dateutil.tz import (
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.compat import PY310
from pandas.errors import OutOfBoundsDatetime
from pandas import (
@pytest.mark.parametrize('tz', [pytz.timezone('US/Eastern'), gettz('US/Eastern'), 'US/Eastern', 'dateutil/US/Eastern'])
def test_timestamp_constructed_by_date_and_tz(self, tz):
    result = Timestamp(date(2012, 3, 11), tz=tz)
    expected = Timestamp('3/11/2012', tz=tz)
    assert result.hour == expected.hour
    assert result == expected