import numpy as np
import pytest
from scipy.conftest import array_api_compatible
from scipy._lib._array_api import (
import scipy._lib.array_api_compat.array_api_compat.numpy as np_compat
@array_api_compatible
def test_check_scalar(self, xp):
    if not is_numpy(xp):
        pytest.skip('Scalars only exist in NumPy')
    if is_numpy(xp):
        with pytest.raises(AssertionError, match='Types do not match.'):
            xp_assert_equal(xp.asarray(0.0), xp.float64(0))
        xp_assert_equal(xp.float64(0), xp.asarray(0.0))