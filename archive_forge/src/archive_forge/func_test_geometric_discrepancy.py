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
def test_geometric_discrepancy(self):
    sample = np.array([[0, 0], [1, 1]])
    assert_allclose(qmc.geometric_discrepancy(sample), np.sqrt(2))
    assert_allclose(qmc.geometric_discrepancy(sample, method='mst'), np.sqrt(2))
    sample = np.array([[0, 0], [0, 1], [0.5, 1]])
    assert_allclose(qmc.geometric_discrepancy(sample), 0.5)
    assert_allclose(qmc.geometric_discrepancy(sample, method='mst'), 0.75)
    sample = np.array([[0, 0], [0.25, 0.25], [1, 1]])
    assert_allclose(qmc.geometric_discrepancy(sample), np.sqrt(2) / 4)
    assert_allclose(qmc.geometric_discrepancy(sample, method='mst'), np.sqrt(2) / 2)
    assert_allclose(qmc.geometric_discrepancy(sample, metric='chebyshev'), 0.25)
    assert_allclose(qmc.geometric_discrepancy(sample, method='mst', metric='chebyshev'), 0.5)
    rng = np.random.default_rng(191468432622931918890291693003068437394)
    sample = qmc.LatinHypercube(d=3, seed=rng).random(50)
    assert_allclose(qmc.geometric_discrepancy(sample), 0.05106012076093356)
    assert_allclose(qmc.geometric_discrepancy(sample, method='mst'), 0.19704396643366182)