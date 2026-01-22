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
def test_interval_array_from_tuple():
    data = [None, (1, 2, -3)]
    arr = pa.array(data, pa.month_day_nano_interval())
    assert isinstance(arr, pa.MonthDayNanoIntervalArray)
    assert arr.type == pa.month_day_nano_interval()
    expected_list = [None, pa.MonthDayNano([1, 2, -3])]
    expected = pa.array(expected_list)
    assert arr.equals(expected)
    assert arr.to_pylist() == expected_list