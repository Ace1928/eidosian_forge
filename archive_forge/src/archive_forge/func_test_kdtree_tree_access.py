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
def test_kdtree_tree_access():
    np.random.seed(1234)
    points = np.random.rand(100, 4)
    t = KDTree(points)
    root = t.tree
    assert isinstance(root, KDTree.innernode)
    assert root.children == points.shape[0]
    nodes = [root]
    while nodes:
        n = nodes.pop(-1)
        if isinstance(n, KDTree.leafnode):
            assert isinstance(n.children, int)
            assert n.children == len(n.idx)
            assert_array_equal(points[n.idx], n._node.data_points)
        else:
            assert isinstance(n, KDTree.innernode)
            assert isinstance(n.split_dim, int)
            assert 0 <= n.split_dim < t.m
            assert isinstance(n.split, float)
            assert isinstance(n.children, int)
            assert n.children == n.less.children + n.greater.children
            nodes.append(n.greater)
            nodes.append(n.less)