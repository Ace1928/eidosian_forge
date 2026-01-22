import sys
import platform
import pytest
import numpy as np
import numpy.core._multiarray_umath as ncu
from numpy.testing import (
def test_simple_conjugate(self):
    ref = np.conj(np.sqrt(complex(1, 1)))

    def f(z):
        return np.sqrt(np.conj(z))
    check_complex_value(f, 1, 1, ref.real, ref.imag, False)