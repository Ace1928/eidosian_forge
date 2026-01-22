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
def test_nonfinite_inputs_gh_18223():
    rng = np.random.default_rng(12345)
    coords = rng.uniform(size=(100, 3), low=0.0, high=0.1)
    t = KDTree(coords, balanced_tree=False, compact_nodes=False)
    bad_coord = [np.nan for _ in range(3)]
    with pytest.raises(ValueError, match='must be finite'):
        t.query(bad_coord)
    with pytest.raises(ValueError, match='must be finite'):
        t.query_ball_point(bad_coord, 1)
    coords[0, :] = np.nan
    with pytest.raises(ValueError, match='must be finite'):
        KDTree(coords, balanced_tree=True, compact_nodes=False)
    with pytest.raises(ValueError, match='must be finite'):
        KDTree(coords, balanced_tree=False, compact_nodes=True)
    with pytest.raises(ValueError, match='must be finite'):
        KDTree(coords, balanced_tree=True, compact_nodes=True)
    with pytest.raises(ValueError, match='must be finite'):
        KDTree(coords, balanced_tree=False, compact_nodes=False)