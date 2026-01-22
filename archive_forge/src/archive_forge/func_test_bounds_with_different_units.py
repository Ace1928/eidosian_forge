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
@pytest.mark.skip_ubsan
def test_bounds_with_different_units(self):
    out_of_bounds_dates = ('1677-09-21', '2262-04-12')
    time_units = ('D', 'h', 'm', 's', 'ms', 'us')
    for date_string in out_of_bounds_dates:
        for unit in time_units:
            dt64 = np.datetime64(date_string, unit)
            ts = Timestamp(dt64)
            if unit in ['s', 'ms', 'us']:
                assert ts._value == dt64.view('i8')
            else:
                assert ts._creso == NpyDatetimeUnit.NPY_FR_s.value
    info = np.iinfo(np.int64)
    msg = 'Out of bounds second timestamp:'
    for value in [info.min + 1, info.max]:
        for unit in ['D', 'h', 'm']:
            dt64 = np.datetime64(value, unit)
            with pytest.raises(OutOfBoundsDatetime, match=msg):
                Timestamp(dt64)
    in_bounds_dates = ('1677-09-23', '2262-04-11')
    for date_string in in_bounds_dates:
        for unit in time_units:
            dt64 = np.datetime64(date_string, unit)
            Timestamp(dt64)