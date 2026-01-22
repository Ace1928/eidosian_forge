import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose, suppress_warnings
from scipy.special._ufuncs import _sinpi as sinpi
from scipy.special._ufuncs import _cospi as cospi
def test_zero_sign():
    y = sinpi(-0.0)
    assert y == 0.0
    assert np.signbit(y)
    y = sinpi(0.0)
    assert y == 0.0
    assert not np.signbit(y)
    y = cospi(0.5)
    assert y == 0.0
    assert not np.signbit(y)