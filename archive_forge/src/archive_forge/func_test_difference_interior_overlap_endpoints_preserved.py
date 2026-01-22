from datetime import (
from hypothesis import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_difference_interior_overlap_endpoints_preserved(self):
    left = RangeIndex(range(4))
    right = RangeIndex(range(1, 3))
    result = left.difference(right)
    expected = RangeIndex(0, 4, 3)
    assert expected.tolist() == [0, 3]
    tm.assert_index_equal(result, expected, exact=True)