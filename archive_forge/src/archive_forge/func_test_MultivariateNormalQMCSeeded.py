import os
from collections import Counter
from itertools import combinations, product
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
from scipy.spatial import distance
from scipy.stats import shapiro
from scipy.stats._sobol import _test_find_index
from scipy.stats import qmc
from scipy.stats._qmc import (
def test_MultivariateNormalQMCSeeded(self):
    rng = np.random.default_rng(180182791534511062935571481899241825000)
    a = rng.standard_normal((2, 2))
    A = a @ a.transpose() + np.diag(rng.random(2))
    engine = qmc.MultivariateNormalQMC(np.array([0, 0]), A, inv_transform=False, seed=rng)
    samples = engine.random(n=2)
    samples_expected = np.array([[-0.64419, -0.882413], [0.837199, 2.045301]])
    assert_allclose(samples, samples_expected, atol=0.0001)
    rng = np.random.default_rng(180182791534511062935571481899241825000)
    a = rng.standard_normal((3, 3))
    A = a @ a.transpose() + np.diag(rng.random(3))
    engine = qmc.MultivariateNormalQMC(np.array([0, 0, 0]), A, inv_transform=False, seed=rng)
    samples = engine.random(n=2)
    samples_expected = np.array([[-0.693853, -1.265338, -0.088024], [1.620193, 2.679222, 0.457343]])
    assert_allclose(samples, samples_expected, atol=0.0001)