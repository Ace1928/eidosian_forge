from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
@pytest.mark.pandas
def test_interval_array_from_dateoffset():
    from pandas.tseries.offsets import DateOffset
    data = [None, DateOffset(years=1, months=1, days=1, seconds=1, microseconds=1, minutes=1, hours=1, weeks=1, nanoseconds=1), DateOffset()]
    arr = pa.array(data)
    assert isinstance(arr, pa.MonthDayNanoIntervalArray)
    assert arr.type == pa.month_day_nano_interval()
    expected_list = [None, pa.MonthDayNano([13, 8, 3661000001001]), pa.MonthDayNano([0, 0, 0])]
    expected = pa.array(expected_list)
    assert arr.equals(expected)
    expected_from_pandas = [None, DateOffset(months=13, days=8, microseconds=datetime.timedelta(seconds=1, microseconds=1, minutes=1, hours=1) // datetime.timedelta(microseconds=1), nanoseconds=1), DateOffset(months=0, days=0, microseconds=0, nanoseconds=0)]
    assert arr.to_pandas().tolist() == expected_from_pandas
    actual_list = pa.array([data]).to_pandas().tolist()
    assert len(actual_list) == 1
    assert list(actual_list[0]) == expected_from_pandas