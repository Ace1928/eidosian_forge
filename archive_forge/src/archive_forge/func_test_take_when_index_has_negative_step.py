import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_take_when_index_has_negative_step(self):
    idx = RangeIndex(11, -4, -2, name='foo')
    result = idx.take(np.array([1, 0, -1, -8]))
    expected = Index([9, 11, -3, 11], dtype=np.int64, name='foo')
    tm.assert_index_equal(result, expected)