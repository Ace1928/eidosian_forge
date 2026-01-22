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
def test_constructor_datetime64_with_tz(self):
    dt = np.datetime64('1970-01-01 05:00:00')
    tzstr = 'UTC+05:00'
    ts = Timestamp(dt, tz=tzstr)
    alt = Timestamp(dt).tz_localize(tzstr)
    assert ts == alt
    assert ts.hour == 5