import numpy as np
import pytest
from sklearn.utils._testing import assert_allclose
from sklearn.utils.arrayfuncs import _all_with_any_reduction_axis_1, min_pos
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_min_pos_no_positive(dtype):
    X = np.full(100, -1.0).astype(dtype, copy=False)
    assert min_pos(X) == np.finfo(dtype).max