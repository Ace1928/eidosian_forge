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
def test_MultinomialDistribution(self):
    seed = np.random.default_rng(77797854505813727292048130876699859000)
    p = np.array([0.12, 0.26, 0.05, 0.35, 0.22])
    engine = qmc.MultinomialQMC(p, n_trials=8192, seed=seed)
    draws = engine.random(1)
    assert_allclose(draws / np.sum(draws), np.atleast_2d(p), atol=0.0001)