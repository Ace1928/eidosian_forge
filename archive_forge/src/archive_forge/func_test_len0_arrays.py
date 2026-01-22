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
def test_len0_arrays(kdtree_type):
    np.random.seed(1234)
    X = np.random.rand(10, 2)
    Y = np.random.rand(10, 2)
    tree = kdtree_type(X)
    d, i = tree.query([0.5, 0.5], k=1)
    z = tree.query_ball_point([0.5, 0.5], 0.1 * d)
    assert_array_equal(z, [])
    d, i = tree.query(Y, k=1)
    mind = d.min()
    z = tree.query_ball_point(Y, 0.1 * mind)
    y = np.empty(shape=(10,), dtype=object)
    y.fill([])
    assert_array_equal(y, z)
    other = kdtree_type(Y)
    y = tree.query_ball_tree(other, 0.1 * mind)
    assert_array_equal(10 * [[]], y)
    y = tree.count_neighbors(other, 0.1 * mind)
    assert_(y == 0)
    y = tree.sparse_distance_matrix(other, 0.1 * mind, output_type='dok_matrix')
    assert_array_equal(y == np.zeros((10, 10)), True)
    y = tree.sparse_distance_matrix(other, 0.1 * mind, output_type='coo_matrix')
    assert_array_equal(y == np.zeros((10, 10)), True)
    y = tree.sparse_distance_matrix(other, 0.1 * mind, output_type='dict')
    assert_equal(y, {})
    y = tree.sparse_distance_matrix(other, 0.1 * mind, output_type='ndarray')
    _dtype = [('i', np.intp), ('j', np.intp), ('v', np.float64)]
    res_dtype = np.dtype(_dtype, align=True)
    z = np.empty(shape=(0,), dtype=res_dtype)
    assert_array_equal(y, z)
    d, i = tree.query(X, k=2)
    mind = d[:, -1].min()
    y = tree.query_pairs(0.1 * mind, output_type='set')
    assert_equal(y, set())
    y = tree.query_pairs(0.1 * mind, output_type='ndarray')
    z = np.empty(shape=(0, 2), dtype=np.intp)
    assert_array_equal(y, z)