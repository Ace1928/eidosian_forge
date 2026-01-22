import re
import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('arr', [[1, 2, np.nan, np.nan], [1, np.nan, 2, np.nan], [1, 2, np.nan], [np.nan, 1, 0, 0, np.nan, 2], [np.nan, 0, 0, 1, 2, 1]])
@pytest.mark.parametrize('fill_value', [np.nan, 0, 1])
def test_unique_na_fill(arr, fill_value):
    a = SparseArray(arr, fill_value=fill_value).unique()
    b = pd.Series(arr).unique()
    assert isinstance(a, SparseArray)
    a = np.asarray(a)
    tm.assert_numpy_array_equal(a, b)