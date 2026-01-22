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
@pytest.mark.parametrize('typ', [int, float])
def test_constructor_int_float_with_YM_unit(self, typ):
    val = typ(150)
    ts = Timestamp(val, unit='Y')
    expected = Timestamp('2120-01-01')
    assert ts == expected
    ts = Timestamp(val, unit='M')
    expected = Timestamp('1982-07-01')
    assert ts == expected