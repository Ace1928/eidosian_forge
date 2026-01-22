import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_assert_index_equal_different_inferred_types():
    msg = 'Index are different\n\nAttribute "inferred_type" are different\n\\[left\\]:  mixed\n\\[right\\]: datetime'
    idx1 = Index([NA, np.datetime64('nat')])
    idx2 = Index([NA, NaT])
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2)