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
@pytest.mark.parametrize('tz', ['dateutil/Europe/London'])
@pytest.mark.parametrize('ts_input,fold,value_out', [(datetime(2019, 10, 27, 1, 30, 0, 0), 0, 1572136200000000), (datetime(2019, 10, 27, 1, 30, 0, 0), 1, 1572139800000000)])
def test_timestamp_constructor_adjust_value_for_fold(self, tz, ts_input, fold, value_out):
    ts = Timestamp(ts_input, tz=tz, fold=fold)
    result = ts._value
    expected = value_out
    assert result == expected