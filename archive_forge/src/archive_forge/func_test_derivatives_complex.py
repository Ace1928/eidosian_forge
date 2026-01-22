import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_derivatives_complex(self):
    x, y = (np.array([-1, -1, 0, 1, 1]), np.array([1, 1j, 0, -1, 1j]))
    func = KroghInterpolator(x, y)
    cmplx = func.derivatives(0)
    cmplx2 = KroghInterpolator(x, y.real).derivatives(0) + 1j * KroghInterpolator(x, y.imag).derivatives(0)
    assert_allclose(cmplx, cmplx2, atol=1e-15)