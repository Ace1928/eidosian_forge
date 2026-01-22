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
def test_vertices_2d(self):
    np.random.seed(1234)
    points = np.random.rand(30, 2)
    hull = qhull.ConvexHull(points)
    assert_equal(np.unique(hull.simplices), np.sort(hull.vertices))
    x, y = hull.points[hull.vertices].T
    angle = np.arctan2(y - y.mean(), x - x.mean())
    assert_(np.all(np.diff(np.unwrap(angle)) > 0))