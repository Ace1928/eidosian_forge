import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_index_equal_levels_mismatch():
    msg = "Index are different\n\nIndex levels are different\n\\[left\\]:  1, Index\\(\\[1, 2, 3\\], dtype='int64'\\)\n\\[right\\]: 2, MultiIndex\\(\\[\\('A', 1\\),\n            \\('A', 2\\),\n            \\('B', 3\\),\n            \\('B', 4\\)\\],\n           \\)"
    idx1 = Index([1, 2, 3])
    idx2 = MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 3), ('B', 4)])
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, exact=False)