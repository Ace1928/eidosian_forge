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
def test_array_from_different_numpy_datetime_units_raises():
    data = [None, datetime.datetime(2017, 4, 4, 12, 11, 10), datetime.datetime(2018, 1, 1, 0, 2, 0)]
    s = np.array(data, dtype='datetime64[s]')
    ms = np.array(data, dtype='datetime64[ms]')
    data = list(s[:2]) + list(ms[2:])
    with pytest.raises(pa.ArrowNotImplementedError):
        pa.array(data)