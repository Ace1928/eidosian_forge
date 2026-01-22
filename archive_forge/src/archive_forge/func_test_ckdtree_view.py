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
def test_ckdtree_view():
    np.random.seed(0)
    n = 100
    k = 4
    points = np.random.randn(n, k)
    kdtree = cKDTree(points)

    def recurse_tree(n):
        assert_(isinstance(n, cKDTreeNode))
        if n.split_dim == -1:
            assert_(n.lesser is None)
            assert_(n.greater is None)
            assert_(n.indices.shape[0] <= kdtree.leafsize)
        else:
            recurse_tree(n.lesser)
            recurse_tree(n.greater)
            x = n.lesser.data_points[:, n.split_dim]
            y = n.greater.data_points[:, n.split_dim]
            assert_(x.max() < y.min())
    recurse_tree(kdtree.tree)
    n = kdtree.tree
    assert_array_equal(np.sort(n.indices), range(100))
    assert_array_equal(kdtree.data[n.indices, :], n.data_points)