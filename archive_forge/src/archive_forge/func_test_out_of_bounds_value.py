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
def test_out_of_bounds_value(self):
    one_us = np.timedelta64(1).astype('timedelta64[us]')
    min_ts_us = np.datetime64(Timestamp.min).astype('M8[us]') + one_us
    max_ts_us = np.datetime64(Timestamp.max).astype('M8[us]')
    Timestamp(min_ts_us)
    Timestamp(max_ts_us)
    us_val = NpyDatetimeUnit.NPY_FR_us.value
    assert Timestamp(min_ts_us - one_us)._creso == us_val
    assert Timestamp(max_ts_us + one_us)._creso == us_val
    too_low = np.datetime64('-292277022657-01-27T08:29', 'm')
    too_high = np.datetime64('292277026596-12-04T15:31', 'm')
    msg = 'Out of bounds'
    with pytest.raises(ValueError, match=msg):
        Timestamp(too_low)
    with pytest.raises(ValueError, match=msg):
        Timestamp(too_high)