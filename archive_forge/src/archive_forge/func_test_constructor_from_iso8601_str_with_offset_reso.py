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
def test_constructor_from_iso8601_str_with_offset_reso(self):
    ts = Timestamp('2016-01-01 04:05:06-01:00')
    assert ts.unit == 's'
    ts = Timestamp('2016-01-01 04:05:06.000-01:00')
    assert ts.unit == 'ms'
    ts = Timestamp('2016-01-01 04:05:06.000000-01:00')
    assert ts.unit == 'us'
    ts = Timestamp('2016-01-01 04:05:06.000000001-01:00')
    assert ts.unit == 'ns'