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
def test_query_ball_point_multithreading(kdtree_type):
    np.random.seed(0)
    n = 5000
    k = 2
    points = np.random.randn(n, k)
    T = kdtree_type(points)
    l1 = T.query_ball_point(points, 0.003, workers=1)
    l2 = T.query_ball_point(points, 0.003, workers=64)
    l3 = T.query_ball_point(points, 0.003, workers=-1)
    for i in range(n):
        if l1[i] or l2[i]:
            assert_array_equal(l1[i], l2[i])
    for i in range(n):
        if l1[i] or l3[i]:
            assert_array_equal(l1[i], l3[i])