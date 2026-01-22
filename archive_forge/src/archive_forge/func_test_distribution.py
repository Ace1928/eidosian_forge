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
@pytest.mark.parametrize('scramble', [True])
def test_distribution(self, scramble):
    d = 50
    engine = self.engine(d=d, scramble=scramble)
    sample = engine.random(1024)
    assert_allclose(np.mean(sample, axis=0), np.repeat(0.5, d), atol=0.01)
    assert_allclose(np.percentile(sample, 25, axis=0), np.repeat(0.25, d), atol=0.01)
    assert_allclose(np.percentile(sample, 75, axis=0), np.repeat(0.75, d), atol=0.01)