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
def test_barely_out_of_bounds(self):
    msg = 'Out of bounds nanosecond timestamp: 2262-04-11 23:47:16'
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        Timestamp('2262-04-11 23:47:16.854775808')