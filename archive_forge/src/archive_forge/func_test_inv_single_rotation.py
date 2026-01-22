import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_inv_single_rotation():
    rnd = np.random.RandomState(0)
    p = Rotation.random(random_state=rnd)
    q = p.inv()
    p_mat = p.as_matrix()
    q_mat = q.as_matrix()
    res1 = np.dot(p_mat, q_mat)
    res2 = np.dot(q_mat, p_mat)
    eye = np.eye(3)
    assert_array_almost_equal(res1, eye)
    assert_array_almost_equal(res2, eye)
    x = Rotation.random(num=1, random_state=rnd)
    y = x.inv()
    x_matrix = x.as_matrix()
    y_matrix = y.as_matrix()
    result1 = np.einsum('...ij,...jk->...ik', x_matrix, y_matrix)
    result2 = np.einsum('...ij,...jk->...ik', y_matrix, x_matrix)
    eye3d = np.empty((1, 3, 3))
    eye3d[:] = np.eye(3)
    assert_array_almost_equal(result1, eye3d)
    assert_array_almost_equal(result2, eye3d)