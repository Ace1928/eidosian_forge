import re
import pytest
from pandas import (
import pandas._testing as tm
from pandas.compat import pa_version_under11p0
def test_list_len():
    ser = Series([[1, 2, 3], [4, None], None], dtype=ArrowDtype(pa.list_(pa.int64())))
    actual = ser.list.len()
    expected = Series([3, 2, None], dtype=ArrowDtype(pa.int32()))
    tm.assert_series_equal(actual, expected)