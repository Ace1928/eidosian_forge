import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_from_rotvec_small_angle():
    rotvec = np.array([[0.0005 / np.sqrt(3), -0.0005 / np.sqrt(3), 0.0005 / np.sqrt(3)], [0.2, 0.3, 0.4], [0, 0, 0]])
    quat = Rotation.from_rotvec(rotvec).as_quat()
    assert_allclose(quat[0, 3], 1)
    assert_allclose(quat[0, :3], rotvec[0] * 0.5)
    assert_allclose(quat[1, 3], 0.9639685)
    assert_allclose(quat[1, :3], np.array([0.09879603932153465, 0.14819405898230198, 0.1975920786430693]))
    assert_equal(quat[2], np.array([0, 0, 0, 1]))