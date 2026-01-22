import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_duplicated_arrow_dtype(self):
    pytest.importorskip('pyarrow')
    ser = Series([True, False, None, False], dtype='bool[pyarrow]')
    result = ser.drop_duplicates()
    expected = Series([True, False, None], dtype='bool[pyarrow]')
    tm.assert_series_equal(result, expected)