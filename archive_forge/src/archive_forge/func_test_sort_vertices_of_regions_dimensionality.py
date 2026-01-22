import numpy as np
import itertools
from numpy.testing import (assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma
def test_sort_vertices_of_regions_dimensionality(self):
    points = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0.5, 0.5, 0.5, 0.5]])
    with pytest.raises(TypeError, match='three-dimensional'):
        sv = SphericalVoronoi(points)
        sv.sort_vertices_of_regions()