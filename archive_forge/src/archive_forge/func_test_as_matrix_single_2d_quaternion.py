import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_as_matrix_single_2d_quaternion():
    quat = [[0, 0, 1, 1]]
    mat = Rotation.from_quat(quat).as_matrix()
    assert_equal(mat.shape, (1, 3, 3))
    expected_mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert_array_almost_equal(mat[0], expected_mat)