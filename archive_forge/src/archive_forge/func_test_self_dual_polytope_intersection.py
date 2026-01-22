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
def test_self_dual_polytope_intersection(self):
    fname = os.path.join(os.path.dirname(__file__), 'data', 'selfdual-4d-polytope.txt')
    ineqs = np.genfromtxt(fname)
    halfspaces = -np.hstack((ineqs[:, 1:], ineqs[:, :1]))
    feas_point = np.array([0.0, 0.0, 0.0, 0.0])
    hs = qhull.HalfspaceIntersection(halfspaces, feas_point)
    assert_equal(hs.intersections.shape, (24, 4))
    assert_almost_equal(hs.dual_volume, 32.0)
    assert_equal(len(hs.dual_facets), 24)
    for facet in hs.dual_facets:
        assert_equal(len(facet), 6)
    dists = halfspaces[:, -1] + halfspaces[:, :-1].dot(feas_point)
    self.assert_unordered_allclose((halfspaces[:, :-1].T / dists).T, hs.dual_points)
    points = itertools.permutations([0.0, 0.0, 0.5, -0.5])
    for point in points:
        assert_equal(np.sum((hs.intersections == point).all(axis=1)), 1)