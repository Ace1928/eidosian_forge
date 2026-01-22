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
def test_kdtree_box_upper_bounds(kdtree_type):
    data = np.linspace(0, 2, 10).reshape(-1, 2)
    data[:, 1] += 10
    with pytest.raises(ValueError):
        kdtree_type(data, leafsize=1, boxsize=1.0)
    with pytest.raises(ValueError):
        kdtree_type(data, leafsize=1, boxsize=(0.0, 2.0))
    kdtree_type(data, leafsize=1, boxsize=(2.0, 0.0))