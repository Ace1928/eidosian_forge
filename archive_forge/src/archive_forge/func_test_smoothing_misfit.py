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
def test_smoothing_misfit(self, kernel):
    rng = np.random.RandomState(0)
    seq = Halton(1, scramble=False, seed=rng)
    noise = 0.2
    rmse_tol = 0.1
    smoothing_range = 10 ** np.linspace(-4, 1, 20)
    x = 3 * seq.random(100)
    y = _1d_test_function(x) + rng.normal(0.0, noise, (100,))
    ytrue = _1d_test_function(x)
    rmse_within_tol = False
    for smoothing in smoothing_range:
        ysmooth = self.build(x, y, epsilon=1.0, smoothing=smoothing, kernel=kernel)(x)
        rmse = np.sqrt(np.mean((ysmooth - ytrue) ** 2))
        if rmse < rmse_tol:
            rmse_within_tol = True
            break
    assert rmse_within_tol