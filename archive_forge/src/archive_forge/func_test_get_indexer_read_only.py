import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
def test_get_indexer_read_only(self):
    idx = interval_range(start=0, end=5)
    arr = np.array([1, 2])
    arr.flags.writeable = False
    result = idx.get_indexer(arr)
    expected = np.array([0, 1])
    tm.assert_numpy_array_equal(result, expected, check_dtype=False)
    result = idx.get_indexer_non_unique(arr)[0]
    tm.assert_numpy_array_equal(result, expected, check_dtype=False)