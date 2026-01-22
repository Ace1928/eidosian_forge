import math
import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
def test_basic_euler():
    zr = 0.05
    yr = -0.4
    xr = 0.2
    M = nea.euler2mat(zr, yr, xr)
    M1 = nea.euler2mat(zr)
    M2 = nea.euler2mat(0, yr)
    M3 = nea.euler2mat(0, 0, xr)
    assert is_valid_rotation(M)
    assert is_valid_rotation(M1)
    assert is_valid_rotation(M2)
    assert is_valid_rotation(M3)
    assert np.allclose(M, np.dot(M3, np.dot(M2, M1)))
    assert np.all(nea.euler2mat(zr) == nea.euler2mat(z=zr))
    assert np.all(nea.euler2mat(0, yr) == nea.euler2mat(y=yr))
    assert np.all(nea.euler2mat(0, 0, xr) == nea.euler2mat(x=xr))
    assert np.allclose(nea.euler2mat(x=-xr), np.linalg.inv(nea.euler2mat(x=xr)))