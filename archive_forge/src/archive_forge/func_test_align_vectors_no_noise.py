import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_align_vectors_no_noise():
    rnd = np.random.RandomState(0)
    c = Rotation.random(random_state=rnd)
    b = rnd.normal(size=(5, 3))
    a = c.apply(b)
    est, rssd = Rotation.align_vectors(a, b)
    assert_allclose(c.as_quat(), est.as_quat())
    assert_allclose(rssd, 0, atol=1e-07)