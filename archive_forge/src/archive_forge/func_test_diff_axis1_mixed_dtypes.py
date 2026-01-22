import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_diff_axis1_mixed_dtypes(self):
    df = DataFrame({'A': range(3), 'B': 2 * np.arange(3, dtype=np.float64)})
    expected = DataFrame({'A': [np.nan, np.nan, np.nan], 'B': df['B'] / 2})
    result = df.diff(axis=1)
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'a': np.arange(3, dtype='float32'), 'b': np.arange(3, dtype='float64')})
    result = df.diff(axis=1)
    expected = DataFrame({'a': df['a'] * np.nan, 'b': df['b'] * 0})
    tm.assert_frame_equal(result, expected)