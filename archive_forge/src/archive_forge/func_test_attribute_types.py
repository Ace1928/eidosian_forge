import numpy as np
import itertools
from numpy.testing import (assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma
@pytest.mark.parametrize('radius', [1, 1.0])
@pytest.mark.parametrize('center', [None, (1, 2, 3), (1.0, 2.0, 3.0)])
def test_attribute_types(self, radius, center):
    points = radius * self.points
    if center is not None:
        points += center
    sv = SphericalVoronoi(points, radius=radius, center=center)
    assert sv.points.dtype is np.dtype(np.float64)
    assert sv.center.dtype is np.dtype(np.float64)
    assert isinstance(sv.radius, float)