import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_diff_axis1_mixed_dtypes_negative_periods(self):
    df = DataFrame({'A': range(3), 'B': 2 * np.arange(3, dtype=np.float64)})
    expected = DataFrame({'A': -1.0 * df['A'], 'B': df['B'] * np.nan})
    result = df.diff(axis=1, periods=-1)
    tm.assert_frame_equal(result, expected)