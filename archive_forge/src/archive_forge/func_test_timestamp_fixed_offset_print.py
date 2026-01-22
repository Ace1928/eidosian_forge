import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util
def test_timestamp_fixed_offset_print():
    pytest.importorskip('pytz')
    arr = pa.array([0], pa.timestamp('s', tz='+02:00'))
    assert str(arr[0]) == '1970-01-01 02:00:00+02:00'