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
def test_kdtree_query_pairs(kdtree_type):
    np.random.seed(0)
    n = 50
    k = 2
    r = 0.1
    r2 = r ** 2
    points = np.random.randn(n, k)
    T = kdtree_type(points)
    brute = set()
    for i in range(n):
        for j in range(i + 1, n):
            v = points[i, :] - points[j, :]
            if np.dot(v, v) <= r2:
                brute.add((i, j))
    l0 = sorted(brute)
    s = T.query_pairs(r)
    l1 = sorted(s)
    assert_array_equal(l0, l1)
    s = T.query_pairs(r, output_type='set')
    l1 = sorted(s)
    assert_array_equal(l0, l1)
    s = set()
    arr = T.query_pairs(r, output_type='ndarray')
    for i in range(arr.shape[0]):
        s.add((int(arr[i, 0]), int(arr[i, 1])))
    l2 = sorted(s)
    assert_array_equal(l0, l2)