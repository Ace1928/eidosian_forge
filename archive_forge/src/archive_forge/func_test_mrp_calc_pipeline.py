import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_mrp_calc_pipeline():
    actual_mrp = np.array([[0, 0, 0], [1, -1, 2], [0.41421356, 0, 0], [0.1, 0.2, 0.1]])
    expected_mrp = np.array([[0, 0, 0], [-0.16666667, 0.16666667, -0.33333333], [0.41421356, 0, 0], [0.1, 0.2, 0.1]])
    assert_allclose(Rotation.from_mrp(actual_mrp).as_mrp(), expected_mrp)