import copy
import numpy as np
import pytest
import pandas as pd
from pandas import Timestamp
import pandas._testing as tm
def test_assert_numpy_array_equal_value_mismatch6():
    msg = 'numpy array are different\n\nnumpy array values are different \\(25\\.0 %\\)\n\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\]\\]\n\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\]\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([[1, 2], [3, 4]]), np.array([[1, 3], [3, 4]]))