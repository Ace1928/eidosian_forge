import numpy as np
import itertools
from numpy.testing import (assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma
def test_vertices_regions_scaling_invariance(self):
    sv_unit = SphericalVoronoi(self.points)
    sv_scaled = SphericalVoronoi(self.points * 2, 2)
    assert_equal(sv_unit.regions, sv_scaled.regions)
    assert_array_almost_equal(sv_unit.vertices * 2, sv_scaled.vertices)