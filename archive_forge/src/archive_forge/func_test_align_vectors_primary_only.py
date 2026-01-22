import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_align_vectors_primary_only():
    atol = 1e-12
    mats_a = Rotation.random(100, random_state=0).as_matrix()
    mats_b = Rotation.random(100, random_state=1).as_matrix()
    for mat_a, mat_b in zip(mats_a, mats_b):
        a = mat_a[0]
        b = mat_b[0]
        R, rssd = Rotation.align_vectors(a, b)
        assert_allclose(R.apply(b), a, atol=atol)
        assert np.isclose(rssd, 0, atol=atol)