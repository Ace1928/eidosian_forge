import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_as_matrix_from_generic_input():
    quats = [[0, 0, 1, 1], [0, 1, 0, 1], [1, 2, 3, 4]]
    mat = Rotation.from_quat(quats).as_matrix()
    assert_equal(mat.shape, (3, 3, 3))
    expected0 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert_array_almost_equal(mat[0], expected0)
    expected1 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    assert_array_almost_equal(mat[1], expected1)
    expected2 = np.array([[0.4, -2, 2.2], [2.8, 1, 0.4], [-1, 2, 2]]) / 3
    assert_array_almost_equal(mat[2], expected2)