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
@pytest.mark.parametrize('arg', ['001-01-01', '0001-01-01'])
def test_out_of_bounds_string_consistency(self, arg):
    msg = "Cannot cast 0001-01-01 00:00:00 to unit='ns' without overflow"
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        Timestamp(arg).as_unit('ns')
    ts = Timestamp(arg)
    assert ts.unit == 's'
    assert ts.year == ts.month == ts.day == 1