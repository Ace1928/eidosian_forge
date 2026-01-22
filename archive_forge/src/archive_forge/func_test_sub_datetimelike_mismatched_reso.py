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
def test_sub_datetimelike_mismatched_reso(self, ts_tz):
    ts = ts_tz
    unit = {NpyDatetimeUnit.NPY_FR_us.value: 'ms', NpyDatetimeUnit.NPY_FR_ms.value: 's', NpyDatetimeUnit.NPY_FR_s.value: 'us'}[ts._creso]
    other = ts.as_unit(unit)
    assert other._creso != ts._creso
    result = ts - other
    assert isinstance(result, Timedelta)
    assert result._value == 0
    assert result._creso == max(ts._creso, other._creso)
    result = other - ts
    assert isinstance(result, Timedelta)
    assert result._value == 0
    assert result._creso == max(ts._creso, other._creso)
    if ts._creso < other._creso:
        other2 = other + Timedelta._from_value_and_reso(1, other._creso)
        exp = ts.as_unit(other.unit) - other2
        res = ts - other2
        assert res == exp
        assert res._creso == max(ts._creso, other._creso)
        res = other2 - ts
        assert res == -exp
        assert res._creso == max(ts._creso, other._creso)
    else:
        ts2 = ts + Timedelta._from_value_and_reso(1, ts._creso)
        exp = ts2 - other.as_unit(ts2.unit)
        res = ts2 - other
        assert res == exp
        assert res._creso == max(ts._creso, other._creso)
        res = other - ts2
        assert res == -exp
        assert res._creso == max(ts._creso, other._creso)