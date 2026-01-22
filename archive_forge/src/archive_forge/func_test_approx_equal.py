import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_approx_equal():
    rng = np.random.RandomState(0)
    p = Rotation.random(10, random_state=rng)
    q = Rotation.random(10, random_state=rng)
    r = p * q.inv()
    r_mag = r.magnitude()
    atol = np.median(r_mag)
    assert_equal(p.approx_equal(q, atol), r_mag < atol)