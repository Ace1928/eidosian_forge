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
@pytest.mark.parametrize('scramble', scramble, ids=ids)
def test_0dim(self, scramble):
    engine = self.engine(d=0, scramble=scramble)
    sample = engine.random(4)
    assert_array_equal(np.empty((4, 0)), sample)