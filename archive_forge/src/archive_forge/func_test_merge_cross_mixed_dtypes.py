import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import (
def test_merge_cross_mixed_dtypes():
    left = DataFrame(['a', 'b', 'c'], columns=['A'])
    right = DataFrame(range(2), columns=['B'])
    result = merge(left, right, how='cross')
    expected = DataFrame({'A': ['a', 'a', 'b', 'b', 'c', 'c'], 'B': [0, 1, 0, 1, 0, 1]})
    tm.assert_frame_equal(result, expected)