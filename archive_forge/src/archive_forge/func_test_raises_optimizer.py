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
def test_raises_optimizer(self):
    message = "'toto' is not a valid optimization method"
    with pytest.raises(ValueError, match=message):
        self.engine(d=1, scramble=False, optimization='toto')