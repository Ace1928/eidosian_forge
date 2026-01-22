import collections
import datetime
import decimal
import itertools
import math
import re
import sys
import hypothesis as h
import numpy as np
import pytest
from pyarrow.pandas_compat import _pandas_api  # noqa
import pyarrow as pa
from pyarrow.tests import util
import pyarrow.tests.strategies as past
@pytest.mark.parametrize(('time_type', 'unit', 'int_type'), [(pa.time32, 's', 'int32'), (pa.time32, 'ms', 'int32'), (pa.time64, 'us', 'int64'), (pa.time64, 'ns', 'int64')])
def test_sequence_time_with_timezone(time_type, unit, int_type):

    def expected_integer_value(t):
        units = ['s', 'ms', 'us', 'ns']
        multiplier = 10 ** (units.index(unit) * 3)
        if t is None:
            return None
        seconds = t.hour * 3600 + t.minute * 60 + t.second + t.microsecond * 10 ** (-6)
        return int(seconds * multiplier)

    def expected_time_value(t):
        if unit == 's':
            return t.replace(microsecond=0)
        elif unit == 'ms':
            return t.replace(microsecond=t.microsecond // 1000 * 1000)
        else:
            return t
    data = [datetime.time(8, 23, 34, 123456), datetime.time(5, 0, 0, 1000), None, datetime.time(1, 11, 56, 432539), datetime.time(23, 10, 0, 437699)]
    ty = time_type(unit)
    arr = pa.array(data, type=ty)
    assert len(arr) == 5
    assert arr.type == ty
    assert arr.null_count == 1
    values = arr.cast(int_type)
    expected = list(map(expected_integer_value, data))
    assert values.to_pylist() == expected
    assert arr[0].as_py() == expected_time_value(data[0])
    assert arr[1].as_py() == expected_time_value(data[1])
    assert arr[2].as_py() is None
    assert arr[3].as_py() == expected_time_value(data[3])
    assert arr[4].as_py() == expected_time_value(data[4])

    def tz(hours, minutes=0):
        offset = datetime.timedelta(hours=hours, minutes=minutes)
        return datetime.timezone(offset)