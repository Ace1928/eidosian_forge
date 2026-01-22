import copy
import numpy as np
import pytest
import pandas as pd
from pandas import Timestamp
import pandas._testing as tm
def test_numpy_array_equal_object():
    a = np.array([Timestamp('2011-01-01'), Timestamp('2011-01-01')])
    b = np.array([Timestamp('2011-01-01'), Timestamp('2011-01-02')])
    msg = 'numpy array are different\n\nnumpy array values are different \\(50\\.0 %\\)\n\\[left\\]:  \\[2011-01-01 00:00:00, 2011-01-01 00:00:00\\]\n\\[right\\]: \\[2011-01-01 00:00:00, 2011-01-02 00:00:00\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(a, b)