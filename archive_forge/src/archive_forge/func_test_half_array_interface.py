import platform
import pytest
import numpy as np
from numpy import uint16, float16, float32, float64
from numpy.testing import assert_, assert_equal, _OLD_PROMOTION, IS_WASM
def test_half_array_interface(self):
    """Test that half is compatible with __array_interface__"""

    class Dummy:
        pass
    a = np.ones((1,), dtype=float16)
    b = Dummy()
    b.__array_interface__ = a.__array_interface__
    c = np.array(b)
    assert_(c.dtype == float16)
    assert_equal(a, c)