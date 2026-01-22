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
def test_ckdtree_parallel(kdtree_type, monkeypatch):
    np.random.seed(0)
    n = 5000
    k = 4
    points = np.random.randn(n, k)
    T = kdtree_type(points)
    T1 = T.query(points, k=5, workers=64)[-1]
    T2 = T.query(points, k=5, workers=-1)[-1]
    T3 = T.query(points, k=5)[-1]
    assert_array_equal(T1, T2)
    assert_array_equal(T1, T3)
    monkeypatch.setattr(os, 'cpu_count', lambda: None)
    with pytest.raises(NotImplementedError, match='Cannot determine the'):
        T.query(points, 1, workers=-1)