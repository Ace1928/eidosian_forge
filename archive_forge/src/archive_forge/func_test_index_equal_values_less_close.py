import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_index_equal_values_less_close(check_exact, rtol):
    idx1 = Index([1, 2, 3.0])
    idx2 = Index([1, 2, 3.0001])
    kwargs = {'check_exact': check_exact, 'rtol': rtol}
    if check_exact or rtol < 0.0005:
        msg = "Index are different\n\nIndex values are different \\(33\\.33333 %\\)\n\\[left\\]:  Index\\(\\[1.0, 2.0, 3.0], dtype='float64'\\)\n\\[right\\]: Index\\(\\[1.0, 2.0, 3.0001\\], dtype='float64'\\)"
        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(idx1, idx2, **kwargs)
    else:
        tm.assert_index_equal(idx1, idx2, **kwargs)