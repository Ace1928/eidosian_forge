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
def test_kdtree_build_modes(kdtree_type):
    np.random.seed(0)
    n = 5000
    k = 4
    points = np.random.randn(n, k)
    T1 = kdtree_type(points).query(points, k=5)[-1]
    T2 = kdtree_type(points, compact_nodes=False).query(points, k=5)[-1]
    T3 = kdtree_type(points, balanced_tree=False).query(points, k=5)[-1]
    T4 = kdtree_type(points, compact_nodes=False, balanced_tree=False).query(points, k=5)[-1]
    assert_array_equal(T1, T2)
    assert_array_equal(T1, T3)
    assert_array_equal(T1, T4)