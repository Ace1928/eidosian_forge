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
@pytest.mark.parametrize('result', [Timestamp(datetime(2000, 1, 2, 3, 4, 5, 6), nanosecond=1), Timestamp(year=2000, month=1, day=2, hour=3, minute=4, second=5, microsecond=6, nanosecond=1), Timestamp(year=2000, month=1, day=2, hour=3, minute=4, second=5, microsecond=6, nanosecond=1, tz='UTC'), Timestamp(2000, 1, 2, 3, 4, 5, 6, None, nanosecond=1), Timestamp(2000, 1, 2, 3, 4, 5, 6, tz=pytz.UTC, nanosecond=1)])
def test_constructor_nanosecond(self, result):
    expected = Timestamp(datetime(2000, 1, 2, 3, 4, 5, 6), tz=result.tz)
    expected = expected + Timedelta(nanoseconds=1)
    assert result == expected