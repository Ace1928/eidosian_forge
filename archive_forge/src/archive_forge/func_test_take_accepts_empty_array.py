import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_take_accepts_empty_array(self):
    idx = RangeIndex(1, 4, name='foo')
    result = idx.take(np.array([]))
    expected = Index([], dtype=np.int64, name='foo')
    tm.assert_index_equal(result, expected)
    idx = RangeIndex(0, name='foo')
    result = idx.take(np.array([]))
    expected = Index([], dtype=np.int64, name='foo')
    tm.assert_index_equal(result, expected)