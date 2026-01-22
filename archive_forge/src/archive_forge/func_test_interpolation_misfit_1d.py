import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
@pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
def test_interpolation_misfit_1d(self, kernel):
    seq = Halton(1, scramble=False, seed=np.random.RandomState())
    x = 3 * seq.random(50)
    xitp = 3 * seq.random(50)
    y = _1d_test_function(x)
    ytrue = _1d_test_function(xitp)
    yitp = self.build(x, y, epsilon=5.0, kernel=kernel)(xitp)
    mse = np.mean((yitp - ytrue) ** 2)
    assert mse < 0.0001