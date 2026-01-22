import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_single_identity_invariance():
    n = 10
    p = Rotation.random(n, random_state=0)
    result = p * Rotation.identity()
    assert_array_almost_equal(p.as_quat(), result.as_quat())
    result = result * p.inv()
    assert_array_almost_equal(result.magnitude(), np.zeros(n))