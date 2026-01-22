import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_align_vectors_scaled_weights():
    n = 10
    a = Rotation.random(n, random_state=0).apply([1, 0, 0])
    b = Rotation.random(n, random_state=1).apply([1, 0, 0])
    scale = 2
    est1, rssd1, cov1 = Rotation.align_vectors(a, b, np.ones(n), True)
    est2, rssd2, cov2 = Rotation.align_vectors(a, b, scale * np.ones(n), True)
    assert_allclose(est1.as_matrix(), est2.as_matrix())
    assert_allclose(np.sqrt(scale) * rssd1, rssd2, atol=1e-06)
    assert_allclose(cov1, cov2)