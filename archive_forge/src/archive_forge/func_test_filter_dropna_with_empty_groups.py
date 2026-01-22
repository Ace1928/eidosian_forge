from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_filter_dropna_with_empty_groups():
    data = Series(np.random.default_rng(2).random(9), index=np.repeat([1, 2, 3], 3))
    grouped = data.groupby(level=0)
    result_false = grouped.filter(lambda x: x.mean() > 1, dropna=False)
    expected_false = Series([np.nan] * 9, index=np.repeat([1, 2, 3], 3))
    tm.assert_series_equal(result_false, expected_false)
    result_true = grouped.filter(lambda x: x.mean() > 1, dropna=True)
    expected_true = Series(index=pd.Index([], dtype=int), dtype=np.float64)
    tm.assert_series_equal(result_true, expected_true)