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
def test_scale_errors(self):
    with pytest.raises(ValueError, match='Sample is not a 2D array'):
        space = [0, 1, 0.5]
        qmc.scale(space, l_bounds=-2, u_bounds=6)
    with pytest.raises(ValueError, match='Bounds are not consistent'):
        space = [[0, 0], [1, 1], [0.5, 0.5]]
        bounds = np.array([[-2, 6], [6, 5]])
        qmc.scale(space, l_bounds=bounds[0], u_bounds=bounds[1])
    with pytest.raises(ValueError, match="'l_bounds' and 'u_bounds' must be broadcastable"):
        space = [[0, 0], [1, 1], [0.5, 0.5]]
        l_bounds, u_bounds = ([-2, 0, 2], [6, 5])
        qmc.scale(space, l_bounds=l_bounds, u_bounds=u_bounds)
    with pytest.raises(ValueError, match="'l_bounds' and 'u_bounds' must be broadcastable"):
        space = [[0, 0], [1, 1], [0.5, 0.5]]
        bounds = np.array([[-2, 0, 2], [6, 5, 5]])
        qmc.scale(space, l_bounds=bounds[0], u_bounds=bounds[1])
    with pytest.raises(ValueError, match='Sample is not in unit hypercube'):
        space = [[0, 0], [1, 1.5], [0.5, 0.5]]
        bounds = np.array([[-2, 0], [6, 5]])
        qmc.scale(space, l_bounds=bounds[0], u_bounds=bounds[1])
    with pytest.raises(ValueError, match='Sample is out of bounds'):
        out = [[-2, 0], [6, 5], [8, 2.5]]
        bounds = np.array([[-2, 0], [6, 5]])
        qmc.scale(out, l_bounds=bounds[0], u_bounds=bounds[1], reverse=True)