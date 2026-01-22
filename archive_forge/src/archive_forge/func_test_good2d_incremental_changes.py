import os
import copy
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.spatial._qhull as qhull
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Voronoi
import itertools
@pytest.mark.parametrize('visibility', ['QG4', 'QG-4'])
@pytest.mark.parametrize('new_gen, expected', [(np.array([[0.3, 0.7]]), np.array([False, False, False, False, False], dtype=bool)), (np.array([[0.3, -0.7]]), np.array([False, True, False, False, False], dtype=bool)), (np.array([[0.3, 0.41]]), np.array([False, False, False, True, True], dtype=bool)), (np.array([[0.5, 0.6], [0.6, 0.6]]), np.array([False, False, True, False, False], dtype=bool)), (np.array([[0.3, 0.6 + 1e-16]]), np.array([False, False, False, False, False], dtype=bool))])
def test_good2d_incremental_changes(self, new_gen, expected, visibility):
    points = np.array([[0.2, 0.2], [0.2, 0.4], [0.4, 0.4], [0.4, 0.2], [0.3, 0.6]])
    hull = qhull.ConvexHull(points=points, incremental=True, qhull_options=visibility)
    hull.add_points(new_gen)
    actual = hull.good
    if '-' in visibility:
        expected = np.invert(expected)
    assert_equal(actual, expected)