import copy
import numpy as np
import pytest
import pandas as pd
from pandas import Timestamp
import pandas._testing as tm
def test_assert_numpy_array_equal_shape_mismatch():
    msg = 'numpy array are different\n\nnumpy array shapes are different\n\\[left\\]:  \\(2L*,\\)\n\\[right\\]: \\(3L*,\\)'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([1, 2]), np.array([3, 4, 5]))