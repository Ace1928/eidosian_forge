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
def test_query_ball_point_vector_r(kdtree_type):
    np.random.seed(1234)
    data = np.random.normal(size=(100, 3))
    query = np.random.normal(size=(100, 3))
    tree = kdtree_type(data)
    d = np.random.uniform(0, 0.3, size=len(query))
    rvector = tree.query_ball_point(query, d)
    rscalar = [tree.query_ball_point(qi, di) for qi, di in zip(query, d)]
    for a, b in zip(rvector, rscalar):
        assert_array_equal(sorted(a), sorted(b))