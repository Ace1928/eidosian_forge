import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_apply_multiple_rotations_single_point():
    mat = np.empty((2, 3, 3))
    mat[0] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    mat[1] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    r = Rotation.from_matrix(mat)
    v1 = np.array([1, 2, 3])
    v2 = np.expand_dims(v1, axis=0)
    v_rotated = np.array([[-2, 1, 3], [1, -3, 2]])
    assert_allclose(r.apply(v1), v_rotated)
    assert_allclose(r.apply(v2), v_rotated)
    v_inverse = np.array([[2, -1, 3], [1, 3, -2]])
    assert_allclose(r.apply(v1, inverse=True), v_inverse)
    assert_allclose(r.apply(v2, inverse=True), v_inverse)