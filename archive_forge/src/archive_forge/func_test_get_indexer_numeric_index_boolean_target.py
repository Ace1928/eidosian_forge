import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('idx_dtype', ['int64', 'float64', 'uint64', 'range'])
@pytest.mark.parametrize('method', ['get_indexer', 'get_indexer_non_unique'])
def test_get_indexer_numeric_index_boolean_target(self, method, idx_dtype):
    if idx_dtype == 'range':
        numeric_index = RangeIndex(4)
    else:
        numeric_index = Index(np.arange(4, dtype=idx_dtype))
    other = Index([True, False, True])
    result = getattr(numeric_index, method)(other)
    expected = np.array([-1, -1, -1], dtype=np.intp)
    if method == 'get_indexer':
        tm.assert_numpy_array_equal(result, expected)
    else:
        missing = np.arange(3, dtype=np.intp)
        tm.assert_numpy_array_equal(result[0], expected)
        tm.assert_numpy_array_equal(result[1], missing)