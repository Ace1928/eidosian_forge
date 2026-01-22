import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_index_equal_length_mismatch(check_exact):
    msg = "Index are different\n\nIndex length are different\n\\[left\\]:  3, Index\\(\\[1, 2, 3\\], dtype='int64'\\)\n\\[right\\]: 4, Index\\(\\[1, 2, 3, 4\\], dtype='int64'\\)"
    idx1 = Index([1, 2, 3])
    idx2 = Index([1, 2, 3, 4])
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, check_exact=check_exact)