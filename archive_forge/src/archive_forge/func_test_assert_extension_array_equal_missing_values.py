import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_assert_extension_array_equal_missing_values():
    arr1 = SparseArray([np.nan, 1, 2, np.nan])
    arr2 = SparseArray([np.nan, 1, 2, 3])
    msg = 'ExtensionArray NA mask are different\n\nExtensionArray NA mask values are different \\(25\\.0 %\\)\n\\[left\\]:  \\[True, False, False, True\\]\n\\[right\\]: \\[True, False, False, False\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_extension_array_equal(arr1, arr2)