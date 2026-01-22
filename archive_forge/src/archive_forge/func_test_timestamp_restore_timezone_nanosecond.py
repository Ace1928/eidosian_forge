import datetime
import io
import warnings
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip
def test_timestamp_restore_timezone_nanosecond():
    ty = pa.timestamp('ns', tz='America/New_York')
    arr = pa.array([1000, 2000, 3000], type=ty)
    table = pa.table([arr], names=['f0'])
    ty_us = pa.timestamp('us', tz='America/New_York')
    expected = pa.table([arr.cast(ty_us)], names=['f0'])
    _check_roundtrip(table, expected=expected, version='2.4')