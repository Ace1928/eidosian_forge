import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
def test_angle_axis2quat():
    q = nq.angle_axis2quat(0, [1, 0, 0])
    assert_array_equal(q, [1, 0, 0, 0])
    q = nq.angle_axis2quat(np.pi, [1, 0, 0])
    assert_array_almost_equal(q, [0, 1, 0, 0])
    q = nq.angle_axis2quat(np.pi, [1, 0, 0], True)
    assert_array_almost_equal(q, [0, 1, 0, 0])
    q = nq.angle_axis2quat(np.pi, [2, 0, 0], False)
    assert_array_almost_equal(q, [0, 1, 0, 0])