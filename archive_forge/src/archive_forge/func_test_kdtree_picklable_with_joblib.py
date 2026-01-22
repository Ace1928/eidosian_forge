import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from sklearn.neighbors._kd_tree import KDTree, KDTree32, KDTree64
from sklearn.neighbors.tests.test_ball_tree import get_dataset_for_binary_tree
from sklearn.utils.parallel import Parallel, delayed
@pytest.mark.parametrize('BinarySearchTree', KD_TREE_CLASSES)
def test_kdtree_picklable_with_joblib(BinarySearchTree):
    """Make sure that KDTree queries work when joblib memmaps.

    Non-regression test for #21685 and #21228."""
    rng = np.random.RandomState(0)
    X = rng.random_sample((10, 3))
    tree = BinarySearchTree(X, leaf_size=2)
    Parallel(n_jobs=2, max_nbytes=1)((delayed(tree.query)(data) for data in 2 * [X]))