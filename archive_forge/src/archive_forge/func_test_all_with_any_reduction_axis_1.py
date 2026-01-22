import numpy as np
import pytest
from sklearn.utils._testing import assert_allclose
from sklearn.utils.arrayfuncs import _all_with_any_reduction_axis_1, min_pos
@pytest.mark.parametrize('dtype', [np.int16, np.int32, np.float32, np.float64])
@pytest.mark.parametrize('value', [0, 1.5, -1])
def test_all_with_any_reduction_axis_1(dtype, value):
    X = np.arange(12, dtype=dtype).reshape(3, 4)
    assert not _all_with_any_reduction_axis_1(X, value=value)
    X[1, :] = value
    assert _all_with_any_reduction_axis_1(X, value=value)