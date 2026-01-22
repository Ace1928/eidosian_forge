import numpy as np
import pytest
from pandas.core.dtypes.generic import ABCIndex
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.integer import (
def test_astype_to_larger_numpy():
    a = pd.array([1, 2], dtype='Int32')
    result = a.astype('int64')
    expected = np.array([1, 2], dtype='int64')
    tm.assert_numpy_array_equal(result, expected)
    a = pd.array([1, 2], dtype='UInt32')
    result = a.astype('uint64')
    expected = np.array([1, 2], dtype='uint64')
    tm.assert_numpy_array_equal(result, expected)