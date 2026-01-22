import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('colnames', [[1, 2, 3], [1.0, 2.0, 3.0]])
def test_numba_numeric_colnames(colnames):
    df = DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64), columns=colnames)
    first_col = colnames[0]
    f = lambda x: x[first_col]
    result = df.apply(f, engine='numba', axis=1)
    expected = df.apply(f, engine='python', axis=1)
    tm.assert_series_equal(result, expected)