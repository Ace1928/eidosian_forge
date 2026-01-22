import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_almost_equal
from pytest import raises as assert_raises
from scipy.spatial import procrustes
def test_procrustes_empty_rows_or_cols(self):
    empty = np.array([[]])
    assert_raises(ValueError, procrustes, empty, empty)