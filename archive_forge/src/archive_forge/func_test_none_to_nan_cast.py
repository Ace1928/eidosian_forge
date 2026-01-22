import sys
import numpy as np
from numpy.core._rational_tests import rational
import pytest
from numpy.testing import (
@pytest.mark.parametrize('dtype', np.typecodes['AllFloat'])
def test_none_to_nan_cast(dtype):
    arr = np.zeros(1, dtype=dtype)
    arr[0] = None
    assert np.isnan(arr)[0]
    assert np.isnan(np.array(None, dtype=dtype))[()]
    assert np.isnan(np.array([None], dtype=dtype))[0]
    assert np.isnan(np.array(None).astype(dtype))[()]