import datetime
import io
import warnings
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.parametrize('unit', ['ms', 'us', 'ns'])
def test_timestamp_restore_timezone(unit):
    ty = pa.timestamp(unit, tz='America/New_York')
    arr = pa.array([1, 2, 3], type=ty)
    t = pa.table([arr], names=['f0'])
    _check_roundtrip(t)