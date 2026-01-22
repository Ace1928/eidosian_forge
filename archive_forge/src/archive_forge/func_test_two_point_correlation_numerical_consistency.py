import itertools
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_equal
from sklearn.neighbors._ball_tree import BallTree, BallTree32, BallTree64
from sklearn.utils import check_random_state
from sklearn.utils._testing import _convert_container
from sklearn.utils.validation import check_array
def test_two_point_correlation_numerical_consistency(global_random_seed):
    X_64, X_32, Y_64, Y_32 = get_dataset_for_binary_tree(random_seed=global_random_seed)
    bt_64 = BallTree64(X_64, leaf_size=10)
    bt_32 = BallTree32(X_32, leaf_size=10)
    r = np.linspace(0, 1, 10)
    counts_64 = bt_64.two_point_correlation(Y_64, r=r, dualtree=True)
    counts_32 = bt_32.two_point_correlation(Y_32, r=r, dualtree=True)
    assert_allclose(counts_64, counts_32)