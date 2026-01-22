import copy
import numpy as np
import pytest
import pandas as pd
from pandas import Timestamp
import pandas._testing as tm
def test_numpy_array_equal_different_na():
    a = np.array([np.nan], dtype=object)
    b = np.array([pd.NA], dtype=object)
    msg = 'numpy array are different\n\nnumpy array values are different \\(100.0 %\\)\n\\[left\\]:  \\[nan\\]\n\\[right\\]: \\[<NA>\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(a, b)