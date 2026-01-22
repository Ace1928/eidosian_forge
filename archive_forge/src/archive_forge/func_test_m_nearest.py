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
def test_m_nearest(self):
    x = self.x
    m = self.m
    dd, ii = self.kdtree.query(x, m)
    d = np.amax(dd)
    i = ii[np.argmax(dd)]
    assert_almost_equal(d ** 2, np.sum((x - self.data[i]) ** 2))
    eps = 1e-08
    assert_equal(np.sum(np.sum((self.data - x[np.newaxis, :]) ** 2, axis=1) < d ** 2 + eps), m)