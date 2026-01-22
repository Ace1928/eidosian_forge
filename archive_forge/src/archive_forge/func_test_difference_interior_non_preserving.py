from datetime import (
from hypothesis import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_difference_interior_non_preserving(self):
    idx = Index(range(10))
    other = idx[3:4]
    result = idx.difference(other)
    expected = Index([0, 1, 2, 4, 5, 6, 7, 8, 9])
    tm.assert_index_equal(result, expected, exact=True)
    other = idx[::3]
    result = idx.difference(other)
    expected = Index([1, 2, 4, 5, 7, 8])
    tm.assert_index_equal(result, expected, exact=True)
    obj = Index(range(20))
    other = obj[:10:2]
    result = obj.difference(other)
    expected = Index([1, 3, 5, 7, 9] + list(range(10, 20)))
    tm.assert_index_equal(result, expected, exact=True)
    other = obj[1:11:2]
    result = obj.difference(other)
    expected = Index([0, 2, 4, 6, 8, 10] + list(range(11, 20)))
    tm.assert_index_equal(result, expected, exact=True)