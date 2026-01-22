from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
@pytest.mark.parametrize('dtype', np.typecodes['Integer'])
@pytest.mark.parametrize(['scalar', 'error'], [(np.float64(np.nan), ValueError), (np.array(-1).astype(np.ulonglong)[()], OverflowError)])
def test_scalar_to_int_coerce_does_not_cast(self, dtype, scalar, error):
    """
        Signed integers are currently different in that they do not cast other
        NumPy scalar, but instead use scalar.__int__(). The hardcoded
        exception to this rule is `np.array(scalar, dtype=integer)`.
        """
    dtype = np.dtype(dtype)
    with np.errstate(invalid='ignore'):
        coerced = np.array(scalar, dtype=dtype)
        cast = np.array(scalar).astype(dtype)
    assert_array_equal(coerced, cast)
    with pytest.raises(error):
        np.array([scalar], dtype=dtype)
    with pytest.raises(error):
        cast[()] = scalar