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
def test_distance_matrix():
    m = 10
    n = 11
    k = 4
    np.random.seed(1234)
    xs = np.random.randn(m, k)
    ys = np.random.randn(n, k)
    ds = distance_matrix(xs, ys)
    assert_equal(ds.shape, (m, n))
    for i in range(m):
        for j in range(n):
            assert_almost_equal(minkowski_distance(xs[i], ys[j]), ds[i, j])