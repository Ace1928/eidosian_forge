import platform
import pytest
import numpy as np
from numpy import uint16, float16, float32, float64
from numpy.testing import assert_, assert_equal, _OLD_PROMOTION, IS_WASM
def test_half_values(self):
    """Confirms a small number of known half values"""
    a = np.array([1.0, -1.0, 2.0, -2.0, 0.0999755859375, 0.333251953125, 65504, -65504, 2.0 ** (-14), -2.0 ** (-14), 2.0 ** (-24), -2.0 ** (-24), 0, -1 / 1e309, np.inf, -np.inf])
    b = np.array([15360, 48128, 16384, 49152, 11878, 13653, 31743, 64511, 1024, 33792, 1, 32769, 0, 32768, 31744, 64512], dtype=uint16)
    b.dtype = float16
    assert_equal(a, b)