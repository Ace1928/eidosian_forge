import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose, suppress_warnings
from scipy.special._ufuncs import _sinpi as sinpi
from scipy.special._ufuncs import _cospi as cospi
def test_integer_real_part():
    x = np.arange(-100, 101)
    y = np.hstack((-np.linspace(310, -30, 10), np.linspace(-30, 310, 10)))
    x, y = np.meshgrid(x, y)
    z = x + 1j * y
    res = sinpi(z)
    assert_equal(res.real, 0.0)
    res = cospi(z)
    assert_equal(res.imag, 0.0)