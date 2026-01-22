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
@pytest.mark.parametrize(('unit', 'expected'), [('s', datetime.timedelta(seconds=-2147483000)), ('ms', datetime.timedelta(milliseconds=-2147483000)), ('us', datetime.timedelta(microseconds=-2147483000)), ('ns', datetime.timedelta(microseconds=-2147483))])
def test_duration_array_roundtrip_corner_cases(unit, expected):
    ty = pa.duration(unit)
    arr = pa.array([-2147483000], type=ty)
    restored = pa.array(arr.to_pylist(), type=ty)
    assert arr.equals(restored)
    expected_list = [expected]
    if unit == 'ns':
        try:
            import pandas as pd
        except ImportError:
            pass
        else:
            expected_list = [pd.Timedelta(-2147483000, unit='ns')]
    assert restored.to_pylist() == expected_list