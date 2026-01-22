import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
def test_get_index_non_unique_non_monotonic(self):
    index = IntervalIndex.from_tuples([(0.0, 1.0), (1.0, 2.0), (0.0, 1.0), (1.0, 2.0)])
    result, _ = index.get_indexer_non_unique([Interval(1.0, 2.0)])
    expected = np.array([1, 3], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)