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
def test_against_logic_error_regression(self):
    np.random.seed(0)
    too_many = np.array(np.random.randn(18, 2), dtype=int)
    tree = self.kdtree_type(too_many, balanced_tree=False, compact_nodes=False)
    d = tree.sparse_distance_matrix(tree, 3).toarray()
    assert_array_almost_equal(d, d.T, decimal=14)