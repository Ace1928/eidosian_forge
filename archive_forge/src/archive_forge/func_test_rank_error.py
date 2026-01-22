import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
def test_rank_error(self):
    y = np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    d = np.array([0.0, 0.0, 0.0])
    match = 'does not have full column rank'
    with pytest.raises(LinAlgError, match=match):
        self.build(y, d, kernel='thin_plate_spline')(y)