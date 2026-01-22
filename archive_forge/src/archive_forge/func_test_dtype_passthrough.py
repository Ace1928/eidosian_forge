import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, IS_PYPY
@pytest.mark.parametrize('dtype', [np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64, np.complex64, np.complex128])
def test_dtype_passthrough(self, dtype):
    x = np.arange(5).astype(dtype)
    y = np.from_dlpack(x)
    assert y.dtype == x.dtype
    assert_array_equal(x, y)