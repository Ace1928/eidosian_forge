import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.arrays.boolean import coerce_to_array
def test_coerce_to_numpy_array():
    arr = pd.array([True, False, None], dtype='boolean')
    result = np.array(arr)
    expected = np.array([True, False, pd.NA], dtype='object')
    tm.assert_numpy_array_equal(result, expected)
    arr = pd.array([True, False, True], dtype='boolean')
    result = np.array(arr)
    expected = np.array([True, False, True], dtype='bool')
    tm.assert_numpy_array_equal(result, expected)
    result = np.array(arr, dtype='bool')
    expected = np.array([True, False, True], dtype='bool')
    tm.assert_numpy_array_equal(result, expected)
    arr = pd.array([True, False, None], dtype='boolean')
    msg = "cannot convert to 'bool'-dtype NumPy array with missing values. Specify an appropriate 'na_value' for this dtype."
    with pytest.raises(ValueError, match=msg):
        np.array(arr, dtype='bool')