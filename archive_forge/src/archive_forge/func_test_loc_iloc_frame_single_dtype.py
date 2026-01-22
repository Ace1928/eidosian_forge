import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_loc_iloc_frame_single_dtype(self, data):
    df = pd.DataFrame({'A': data})
    expected = pd.Series([data[2]], index=['A'], name=2, dtype=data.dtype)
    result = df.loc[2]
    tm.assert_series_equal(result, expected)
    expected = pd.Series([data[-1]], index=['A'], name=len(data) - 1, dtype=data.dtype)
    result = df.iloc[-1]
    tm.assert_series_equal(result, expected)