import os
import pytest
from pandas import (
import pandas._testing as tm
def test_is_sorted():
    arr = array([1, 2, 3], dtype='Int64')
    tm.assert_is_sorted(arr)
    arr = array([4, 2, 3], dtype='Int64')
    with pytest.raises(AssertionError, match='ExtensionArray are different'):
        tm.assert_is_sorted(arr)