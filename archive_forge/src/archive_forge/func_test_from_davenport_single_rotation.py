import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_from_davenport_single_rotation():
    axis = [0, 0, 1]
    quat = Rotation.from_davenport(axis, 'extrinsic', 90, degrees=True).as_quat()
    expected_quat = np.array([0, 0, 1, 1]) / np.sqrt(2)
    assert_allclose(quat, expected_quat)