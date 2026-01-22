import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_take_accepts_non_int64_array(self):
    idx = RangeIndex(1, 4, name='foo')
    result = idx.take(np.array([2, 1], dtype=np.uint32))
    expected = Index([3, 2], dtype=np.int64, name='foo')
    tm.assert_index_equal(result, expected)