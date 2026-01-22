import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_from_square_quat_matrix():
    x = np.array([[3, 0, 0, 4], [5, 0, 12, 0], [0, 0, 0, 1], [-1, -1, -1, 1], [0, 0, 0, -1], [-1, -1, -1, -1]])
    r = Rotation.from_quat(x)
    expected_quat = x / np.array([[5], [13], [1], [2], [1], [2]])
    assert_array_almost_equal(r.as_quat(), expected_quat)