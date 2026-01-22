import copy
import numpy as np
import pytest
import pandas as pd
from pandas import Timestamp
import pandas._testing as tm
def test_assert_numpy_array_equal_value_mismatch1():
    msg = 'numpy array are different\n\nnumpy array values are different \\(66\\.66667 %\\)\n\\[left\\]:  \\[nan, 2\\.0, 3\\.0\\]\n\\[right\\]: \\[1\\.0, nan, 3\\.0\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([np.nan, 2, 3]), np.array([1, np.nan, 3]))