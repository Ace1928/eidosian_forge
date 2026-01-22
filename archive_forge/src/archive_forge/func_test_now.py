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
def test_now(self):
    ts_from_string = Timestamp('now')
    ts_from_method = Timestamp.now()
    ts_datetime = datetime.now()
    ts_from_string_tz = Timestamp('now', tz='US/Eastern')
    ts_from_method_tz = Timestamp.now(tz='US/Eastern')
    delta = Timedelta(seconds=1)
    assert abs(ts_from_method - ts_from_string) < delta
    assert abs(ts_datetime - ts_from_method) < delta
    assert abs(ts_from_method_tz - ts_from_string_tz) < delta
    assert abs(ts_from_string_tz.tz_localize(None) - ts_from_method_tz.tz_localize(None)) < delta