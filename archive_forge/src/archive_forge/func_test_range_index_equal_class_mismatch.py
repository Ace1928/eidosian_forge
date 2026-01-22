import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_range_index_equal_class_mismatch(check_exact):
    msg = "Index are different\n\nIndex classes are different\n\\[left\\]:  Index\\(\\[1, 2, 3\\], dtype='int64'\\)\n\\[right\\]: "
    idx1 = Index([1, 2, 3])
    idx2 = RangeIndex(range(3))
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, exact=True, check_exact=check_exact)