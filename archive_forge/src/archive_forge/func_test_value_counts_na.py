import numpy as np
import pytest
from pandas.compat import IS64
import pandas as pd
import pandas._testing as tm
def test_value_counts_na():
    arr = pd.array([0.1, 0.2, 0.1, pd.NA], dtype='Float64')
    result = arr.value_counts(dropna=False)
    idx = pd.Index([0.1, 0.2, pd.NA], dtype=arr.dtype)
    assert idx.dtype == arr.dtype
    expected = pd.Series([2, 1, 1], index=idx, dtype='Int64', name='count')
    tm.assert_series_equal(result, expected)
    result = arr.value_counts(dropna=True)
    expected = pd.Series([2, 1], index=idx[:-1], dtype='Int64', name='count')
    tm.assert_series_equal(result, expected)