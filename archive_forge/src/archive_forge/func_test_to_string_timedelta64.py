from datetime import (
from io import StringIO
import re
import sys
from textwrap import dedent
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_to_string_timedelta64(self):
    Series(np.array([1100, 20], dtype='timedelta64[ns]')).to_string()
    ser = Series(date_range('2012-1-1', periods=3, freq='D'))
    y = ser - ser.shift(1)
    result = y.to_string()
    assert '1 days' in result
    assert '00:00:00' not in result
    assert 'NaT' in result
    o = Series([datetime(2012, 1, 1, microsecond=150)] * 3)
    y = ser - o
    result = y.to_string()
    assert '-1 days +23:59:59.999850' in result
    o = Series([datetime(2012, 1, 1, 1)] * 3)
    y = ser - o
    result = y.to_string()
    assert '-1 days +23:00:00' in result
    assert '1 days 23:00:00' in result
    o = Series([datetime(2012, 1, 1, 1, 1)] * 3)
    y = ser - o
    result = y.to_string()
    assert '-1 days +22:59:00' in result
    assert '1 days 22:59:00' in result
    o = Series([datetime(2012, 1, 1, 1, 1, microsecond=150)] * 3)
    y = ser - o
    result = y.to_string()
    assert '-1 days +22:58:59.999850' in result
    assert '0 days 22:58:59.999850' in result
    td = timedelta(minutes=5, seconds=3)
    s2 = Series(date_range('2012-1-1', periods=3, freq='D')) + td
    y = ser - s2
    result = y.to_string()
    assert '-1 days +23:54:57' in result
    td = timedelta(microseconds=550)
    s2 = Series(date_range('2012-1-1', periods=3, freq='D')) + td
    y = ser - td
    result = y.to_string()
    assert '2012-01-01 23:59:59.999450' in result
    td = Series(timedelta_range('1 days', periods=3))
    result = td.to_string()
    assert result == '0   1 days\n1   2 days\n2   3 days'