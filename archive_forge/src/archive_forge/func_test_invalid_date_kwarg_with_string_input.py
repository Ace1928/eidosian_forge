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
@pytest.mark.parametrize('arg', ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond', 'nanosecond'])
def test_invalid_date_kwarg_with_string_input(self, arg):
    kwarg = {arg: 1}
    msg = 'Cannot pass a date attribute keyword argument'
    with pytest.raises(ValueError, match=msg):
        Timestamp('2010-10-10 12:59:59.999999999', **kwarg)