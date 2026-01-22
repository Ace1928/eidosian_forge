import math
import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
@pytest.mark.parametrize('x, y, z', eg_rots)
def test_quats(x, y, z):
    M1 = nea.euler2mat(z, y, x)
    quatM = nq.mat2quat(M1)
    quat = nea.euler2quat(z, y, x)
    assert nq.nearly_equivalent(quatM, quat)
    quatS = sympy_euler2quat(z, y, x)
    assert nq.nearly_equivalent(quat, quatS)
    zp, yp, xp = nea.quat2euler(quat)
    M2 = nea.euler2mat(zp, yp, xp)
    assert_array_almost_equal(M1, M2)