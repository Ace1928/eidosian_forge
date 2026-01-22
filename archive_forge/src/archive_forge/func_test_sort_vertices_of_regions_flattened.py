import numpy as np
import itertools
from numpy.testing import (assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma
def test_sort_vertices_of_regions_flattened(self):
    expected = sorted([[0, 6, 5, 2, 3], [2, 3, 10, 11, 8, 7], [0, 6, 4, 1], [4, 8, 7, 5, 6], [9, 11, 10], [2, 7, 5], [1, 4, 8, 11, 9], [0, 3, 10, 9, 1]])
    expected = list(itertools.chain(*sorted(expected)))
    sv = SphericalVoronoi(self.points)
    sv.sort_vertices_of_regions()
    actual = list(itertools.chain(*sorted(sv.regions)))
    assert_array_equal(actual, expected)