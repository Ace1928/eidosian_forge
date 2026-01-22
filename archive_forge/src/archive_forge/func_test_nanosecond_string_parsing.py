import calendar
from datetime import (
import locale
import time
import unicodedata
from dateutil.tz import (
from hypothesis import (
import numpy as np
import pytest
import pytz
from pytz import utc
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.timezones import (
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
def test_nanosecond_string_parsing(self):
    ts = Timestamp('2013-05-01 07:15:45.123456789')
    expected_repr = '2013-05-01 07:15:45.123456789'
    expected_value = 1367392545123456789
    assert ts._value == expected_value
    assert expected_repr in repr(ts)
    ts = Timestamp('2013-05-01 07:15:45.123456789+09:00', tz='Asia/Tokyo')
    assert ts._value == expected_value - 9 * 3600 * 1000000000
    assert expected_repr in repr(ts)
    ts = Timestamp('2013-05-01 07:15:45.123456789', tz='UTC')
    assert ts._value == expected_value
    assert expected_repr in repr(ts)
    ts = Timestamp('2013-05-01 07:15:45.123456789', tz='US/Eastern')
    assert ts._value == expected_value + 4 * 3600 * 1000000000
    assert expected_repr in repr(ts)
    ts = Timestamp('20130501T071545.123456789')
    assert ts._value == expected_value
    assert expected_repr in repr(ts)