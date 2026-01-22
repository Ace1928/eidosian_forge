import numpy as np
import itertools
from numpy.testing import (assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma
def test_sort_vertices_of_regions(self):
    sv = SphericalVoronoi(self.points)
    unsorted_regions = sv.regions
    sv.sort_vertices_of_regions()
    assert_equal(sorted(sv.regions), sorted(unsorted_regions))