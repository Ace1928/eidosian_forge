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
def test_points_near_l1(self):
    x = self.x
    d = self.d
    dd, ii = self.kdtree.query(x, k=self.kdtree.n, p=1, distance_upper_bound=d)
    eps = 1e-08
    hits = 0
    for near_d, near_i in zip(dd, ii):
        if near_d == np.inf:
            continue
        hits += 1
        assert_almost_equal(near_d, self.distance(x, self.data[near_i], 1))
        assert_(near_d < d + eps, f'near_d={near_d:g} should be less than {d:g}')
    assert_equal(np.sum(self.distance(self.data, x, 1) < d + eps), hits)