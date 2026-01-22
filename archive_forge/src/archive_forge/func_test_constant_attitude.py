from itertools import product
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises
from scipy.spatial.transform import Rotation, RotationSpline
from scipy.spatial.transform._rotation_spline import (
def test_constant_attitude():
    times = np.arange(10)
    rotations = Rotation.from_rotvec(np.ones((10, 3)))
    spline = RotationSpline(times, rotations)
    times_check = np.linspace(-1, 11)
    assert_allclose(spline(times_check).as_rotvec(), 1, rtol=1e-15)
    assert_allclose(spline(times_check, 1), 0, atol=1e-17)
    assert_allclose(spline(times_check, 2), 0, atol=1e-17)
    assert_allclose(spline(5.5).as_rotvec(), 1, rtol=1e-15)
    assert_allclose(spline(5.5, 1), 0, atol=1e-17)
    assert_allclose(spline(5.5, 2), 0, atol=1e-17)