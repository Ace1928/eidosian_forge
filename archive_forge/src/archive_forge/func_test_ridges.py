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
@pytest.mark.parametrize('name', sorted(DATASETS))
def test_ridges(self, name):
    points = DATASETS[name]
    tree = KDTree(points)
    vor = qhull.Voronoi(points)
    for p, v in vor.ridge_dict.items():
        if not np.all(np.asarray(v) >= 0):
            continue
        ridge_midpoint = vor.vertices[v].mean(axis=0)
        d = 1e-06 * (points[p[0]] - ridge_midpoint)
        dist, k = tree.query(ridge_midpoint + d, k=1)
        assert_equal(k, p[0])
        dist, k = tree.query(ridge_midpoint - d, k=1)
        assert_equal(k, p[1])