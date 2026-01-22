import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_align_vectors_no_rotation():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = x.copy()
    r, rssd = Rotation.align_vectors(x, y)
    assert_array_almost_equal(r.as_matrix(), np.eye(3))
    assert_allclose(rssd, 0, atol=1e-06)