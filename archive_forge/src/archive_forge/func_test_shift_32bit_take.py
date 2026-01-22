import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['int32', 'int64'])
def test_shift_32bit_take(self, frame_or_series, dtype):
    index = date_range('2000-01-01', periods=5)
    arr = np.arange(5, dtype=dtype)
    s1 = frame_or_series(arr, index=index)
    p = arr[1]
    result = s1.shift(periods=p)
    expected = frame_or_series([np.nan, 0, 1, 2, 3], index=index)
    tm.assert_equal(result, expected)