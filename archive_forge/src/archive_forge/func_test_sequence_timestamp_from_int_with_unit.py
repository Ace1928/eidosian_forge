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
@pytest.mark.pandas
@pytest.mark.skipif(sys.platform == 'win32' and (not util.windows_has_tzdata()), reason='Timezone database is not installed on Windows')
def test_sequence_timestamp_from_int_with_unit():
    data = [1]
    s = pa.timestamp('s')
    ms = pa.timestamp('ms')
    us = pa.timestamp('us')
    ns = pa.timestamp('ns')
    arr_s = pa.array(data, type=s)
    assert len(arr_s) == 1
    assert arr_s.type == s
    assert repr(arr_s[0]) == "<pyarrow.TimestampScalar: '1970-01-01T00:00:01'>"
    assert str(arr_s[0]) == '1970-01-01 00:00:01'
    arr_ms = pa.array(data, type=ms)
    assert len(arr_ms) == 1
    assert arr_ms.type == ms
    assert repr(arr_ms[0].as_py()) == 'datetime.datetime(1970, 1, 1, 0, 0, 0, 1000)'
    assert str(arr_ms[0]) == '1970-01-01 00:00:00.001000'
    arr_us = pa.array(data, type=us)
    assert len(arr_us) == 1
    assert arr_us.type == us
    assert repr(arr_us[0].as_py()) == 'datetime.datetime(1970, 1, 1, 0, 0, 0, 1)'
    assert str(arr_us[0]) == '1970-01-01 00:00:00.000001'
    arr_ns = pa.array(data, type=ns)
    assert len(arr_ns) == 1
    assert arr_ns.type == ns
    assert repr(arr_ns[0].as_py()) == "Timestamp('1970-01-01 00:00:00.000000001')"
    assert str(arr_ns[0]) == '1970-01-01 00:00:00.000000001'
    expected_exc = TypeError

    class CustomClass:
        pass
    for ty in [ns, pa.date32(), pa.date64()]:
        with pytest.raises(expected_exc):
            pa.array([1, CustomClass()], type=ty)