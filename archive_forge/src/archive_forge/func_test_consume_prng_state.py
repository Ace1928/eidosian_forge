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
def test_consume_prng_state(self):
    rng = np.random.default_rng(216148415951487386220755762152260027040)
    sample = []
    for i in range(3):
        engine = self.engine(d=2, scramble=True, seed=rng)
        sample.append(engine.random(4))
    with pytest.raises(AssertionError, match='Arrays are not equal'):
        assert_equal(sample[0], sample[1])
    with pytest.raises(AssertionError, match='Arrays are not equal'):
        assert_equal(sample[0], sample[2])