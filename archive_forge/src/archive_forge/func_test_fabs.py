import sys
import platform
import pytest
import numpy as np
import numpy.core._multiarray_umath as ncu
from numpy.testing import (
def test_fabs(self):
    x = np.array([1 + 0j], dtype=complex)
    assert_array_equal(np.abs(x), np.real(x))
    x = np.array([complex(1, np.NZERO)], dtype=complex)
    assert_array_equal(np.abs(x), np.real(x))
    x = np.array([complex(np.inf, np.NZERO)], dtype=complex)
    assert_array_equal(np.abs(x), np.real(x))
    x = np.array([complex(np.nan, np.NZERO)], dtype=complex)
    assert_array_equal(np.abs(x), np.real(x))