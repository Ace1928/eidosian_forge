import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
@pytest.mark.parametrize('dtype', _numeric_dtypes)
@pytest.mark.parametrize('mode', _all_modes.keys())
def test_dtype_persistence(dtype, mode):
    arr = np.zeros((3, 2, 1), dtype=dtype)
    result = np.pad(arr, 1, mode=mode)
    assert result.dtype == dtype