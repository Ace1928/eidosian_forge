import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_int_float_index_equal_class_mismatch(check_exact):
    msg = 'Index are different\n\nAttribute "inferred_type" are different\n\\[left\\]:  integer\n\\[right\\]: floating'
    idx1 = Index([1, 2, 3])
    idx2 = Index([1, 2, 3], dtype=np.float64)
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, exact=True, check_exact=check_exact)