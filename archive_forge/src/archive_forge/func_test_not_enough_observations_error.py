import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
def test_not_enough_observations_error(self):
    y = np.linspace(0, 1, 1)[:, None]
    d = np.zeros(1)
    match = 'At least 2 data points are required'
    with pytest.raises(ValueError, match=match):
        self.build(y, d, kernel='thin_plate_spline')