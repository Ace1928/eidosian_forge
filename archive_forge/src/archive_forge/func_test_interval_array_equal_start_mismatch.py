import pytest
from pandas import interval_range
import pandas._testing as tm
def test_interval_array_equal_start_mismatch():
    kwargs = {'periods': 4}
    arr1 = interval_range(start=0, **kwargs).values
    arr2 = interval_range(start=1, **kwargs).values
    msg = 'IntervalArray.left are different\n\nIntervalArray.left values are different \\(100.0 %\\)\n\\[left\\]:  \\[0, 1, 2, 3\\]\n\\[right\\]: \\[1, 2, 3, 4\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_interval_array_equal(arr1, arr2)