import math
import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
@pytest.mark.parametrize('x, y, z', eg_rots)
def test_euler_mat_2(x, y, z):
    M1 = nea.euler2mat(z, y, x)
    M2 = sympy_euler(z, y, x)
    assert_array_almost_equal(M1, M2)
    M3 = np.dot(x_only(x), np.dot(y_only(y), z_only(z)))
    assert_array_almost_equal(M1, M3)
    zp, yp, xp = nea.mat2euler(M1)
    M4 = nea.euler2mat(zp, yp, xp)
    assert_array_almost_equal(M1, M4)