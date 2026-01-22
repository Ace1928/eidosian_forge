import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_matrix_calculation_pipeline():
    mat = special_ortho_group.rvs(3, size=10, random_state=0)
    assert_array_almost_equal(Rotation.from_matrix(mat).as_matrix(), mat)