import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_values_mixed_dtypes(self, float_frame, float_string_frame):
    frame = float_frame
    arr = frame.values
    frame_cols = frame.columns
    for i, row in enumerate(arr):
        for j, value in enumerate(row):
            col = frame_cols[j]
            if np.isnan(value):
                assert np.isnan(frame[col].iloc[i])
            else:
                assert value == frame[col].iloc[i]
    arr = float_string_frame[['foo', 'A']].values
    assert arr[0, 0] == 'bar'
    df = DataFrame({'complex': [1j, 2j, 3j], 'real': [1, 2, 3]})
    arr = df.values
    assert arr[0, 0] == 1j