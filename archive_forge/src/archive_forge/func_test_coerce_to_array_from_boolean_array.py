import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.arrays.boolean import coerce_to_array
def test_coerce_to_array_from_boolean_array():
    values = np.array([True, False, True, False], dtype='bool')
    mask = np.array([False, False, False, True], dtype='bool')
    arr = BooleanArray(values, mask)
    result = BooleanArray(*coerce_to_array(arr))
    tm.assert_extension_array_equal(result, arr)
    assert result._data is arr._data
    assert result._mask is arr._mask
    result = BooleanArray(*coerce_to_array(arr), copy=True)
    tm.assert_extension_array_equal(result, arr)
    assert result._data is not arr._data
    assert result._mask is not arr._mask
    with pytest.raises(ValueError, match='cannot pass mask for BooleanArray input'):
        coerce_to_array(arr, mask=mask)