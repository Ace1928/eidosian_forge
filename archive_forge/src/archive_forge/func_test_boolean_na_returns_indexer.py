import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.indexers import check_array_indexer
@pytest.mark.parametrize('indexer', [[True, False, None], pd.array([True, False, None], dtype='boolean')])
def test_boolean_na_returns_indexer(indexer):
    arr = np.array([1, 2, 3])
    result = check_array_indexer(arr, indexer)
    expected = np.array([True, False, False], dtype=bool)
    tm.assert_numpy_array_equal(result, expected)