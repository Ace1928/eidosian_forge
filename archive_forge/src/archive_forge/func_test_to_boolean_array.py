import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.arrays.boolean import coerce_to_array
def test_to_boolean_array():
    expected = BooleanArray(np.array([True, False, True]), np.array([False, False, False]))
    result = pd.array([True, False, True], dtype='boolean')
    tm.assert_extension_array_equal(result, expected)
    result = pd.array(np.array([True, False, True]), dtype='boolean')
    tm.assert_extension_array_equal(result, expected)
    result = pd.array(np.array([True, False, True], dtype=object), dtype='boolean')
    tm.assert_extension_array_equal(result, expected)
    expected = BooleanArray(np.array([True, False, True]), np.array([False, False, True]))
    result = pd.array([True, False, None], dtype='boolean')
    tm.assert_extension_array_equal(result, expected)
    result = pd.array(np.array([True, False, None], dtype=object), dtype='boolean')
    tm.assert_extension_array_equal(result, expected)