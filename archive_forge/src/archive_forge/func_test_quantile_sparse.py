import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('df,expected', [[DataFrame({0: Series(pd.arrays.SparseArray([1, 2])), 1: Series(pd.arrays.SparseArray([3, 4]))}), Series([1.5, 3.5], name=0.5)], [DataFrame(Series([0.0, None, 1.0, 2.0], dtype='Sparse[float]')), Series([1.0], name=0.5)]])
def test_quantile_sparse(self, df, expected):
    result = df.quantile()
    expected = expected.astype('Sparse[float]')
    tm.assert_series_equal(result, expected)