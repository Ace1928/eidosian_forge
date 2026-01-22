import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_almost_equal
from pytest import raises as assert_raises
from scipy.spatial import procrustes
def test_procrustes_bad_number_of_dimensions(self):
    assert_raises(ValueError, procrustes, np.array([1, 1, 2, 3, 5, 8]), np.array([[1, 2], [3, 4]]))
    assert_raises(ValueError, procrustes, np.array([1, 1, 2, 3, 5, 8]), np.array([1, 1, 2, 3, 5, 8]))
    assert_raises(ValueError, procrustes, np.array(7), np.array(11))
    assert_raises(ValueError, procrustes, np.array([[[11], [7]]]), np.array([[[5, 13]]]))