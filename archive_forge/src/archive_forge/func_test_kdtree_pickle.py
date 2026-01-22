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
def test_kdtree_pickle(kdtree_type):
    import pickle
    np.random.seed(0)
    n = 50
    k = 4
    points = np.random.randn(n, k)
    T1 = kdtree_type(points)
    tmp = pickle.dumps(T1)
    T2 = pickle.loads(tmp)
    T1 = T1.query(points, k=5)[-1]
    T2 = T2.query(points, k=5)[-1]
    assert_array_equal(T1, T2)