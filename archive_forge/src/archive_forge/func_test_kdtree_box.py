import os
from numpy.testing import (assert_equal, assert_array_equal, assert_,
from pytest import raises as assert_raises
import pytest
from platform import python_implementation
import numpy as np
from scipy.spatial import KDTree, Rectangle, distance_matrix, cKDTree
from scipy.spatial._ckdtree import cKDTreeNode
from scipy.spatial import minkowski_distance
import itertools
def test_kdtree_box(kdtree_type):
    n = 2000
    m = 3
    k = 3
    np.random.seed(1234)
    data = np.random.uniform(size=(n, m))
    kdtree = kdtree_type(data, leafsize=1, boxsize=1.0)
    kdtree2 = kdtree_type(data, leafsize=1)
    for p in [1, 2, 3.0, np.inf]:
        dd, ii = kdtree.query(data, k, p=p)
        dd1, ii1 = kdtree.query(data + 1.0, k, p=p)
        assert_almost_equal(dd, dd1)
        assert_equal(ii, ii1)
        dd1, ii1 = kdtree.query(data - 1.0, k, p=p)
        assert_almost_equal(dd, dd1)
        assert_equal(ii, ii1)
        dd2, ii2 = simulate_periodic_box(kdtree2, data, k, boxsize=1.0, p=p)
        assert_almost_equal(dd, dd2)
        assert_equal(ii, ii2)