import numpy as np
import pytest
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import (
@pytest.mark.parametrize('dtype, val', ([object, 1], [object, 'a'], [float, 1]))
def test_object_dtype_isnan(dtype, val):
    X = np.array([[val, np.nan], [np.nan, val]], dtype=dtype)
    expected_mask = np.array([[False, True], [True, False]])
    mask = _object_dtype_isnan(X)
    assert_array_equal(mask, expected_mask)