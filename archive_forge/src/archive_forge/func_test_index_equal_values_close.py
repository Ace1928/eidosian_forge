import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_index_equal_values_close(check_exact):
    idx1 = Index([1, 2, 3.0])
    idx2 = Index([1, 2, 3.0000000001])
    if check_exact:
        msg = "Index are different\n\nIndex values are different \\(33\\.33333 %\\)\n\\[left\\]:  Index\\(\\[1.0, 2.0, 3.0], dtype='float64'\\)\n\\[right\\]: Index\\(\\[1.0, 2.0, 3.0000000001\\], dtype='float64'\\)"
        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(idx1, idx2, check_exact=check_exact)
    else:
        tm.assert_index_equal(idx1, idx2, check_exact=check_exact)