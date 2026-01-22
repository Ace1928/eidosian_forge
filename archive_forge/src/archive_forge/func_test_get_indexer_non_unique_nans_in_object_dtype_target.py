import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
def test_get_indexer_non_unique_nans_in_object_dtype_target(nulls_fixture):
    idx = Index([1.0, 2.0])
    target = Index([1, nulls_fixture], dtype='object')
    result_idx, result_missing = idx.get_indexer_non_unique(target)
    tm.assert_numpy_array_equal(result_idx, np.array([0, -1], dtype=np.intp))
    tm.assert_numpy_array_equal(result_missing, np.array([1], dtype=np.intp))