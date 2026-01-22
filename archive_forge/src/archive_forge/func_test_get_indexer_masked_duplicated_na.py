import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
def test_get_indexer_masked_duplicated_na(self):
    idx = Index([1, 2, NA, NA], dtype='Int64')
    result = idx.get_indexer_for(Index([1, NA], dtype='Int64'))
    expected = np.array([0, 2, 3], dtype=result.dtype)
    tm.assert_numpy_array_equal(result, expected)