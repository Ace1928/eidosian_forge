import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_almost_equal
from pytest import raises as assert_raises
from scipy.spatial import procrustes
def test_procrustes_no_variation(self):
    assert_raises(ValueError, procrustes, np.array([[42, 42], [42, 42]]), np.array([[45, 45], [45, 45]]))