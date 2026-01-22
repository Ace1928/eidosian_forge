import itertools
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_equal
from sklearn.neighbors._ball_tree import BallTree, BallTree32, BallTree64
from sklearn.utils import check_random_state
from sklearn.utils._testing import _convert_container
from sklearn.utils.validation import check_array
@pytest.mark.parametrize('BallTreeImplementation, decimal_tol', zip(BALL_TREE_CLASSES, [6, 5]))
def test_query_haversine(BallTreeImplementation, decimal_tol):
    rng = check_random_state(0)
    X = 2 * np.pi * rng.random_sample((40, 2))
    bt = BallTreeImplementation(X, leaf_size=1, metric='haversine')
    dist1, ind1 = bt.query(X, k=5)
    dist2, ind2 = brute_force_neighbors(X, X, k=5, metric='haversine')
    assert_array_almost_equal(dist1, dist2, decimal=decimal_tol)
    assert_array_almost_equal(ind1, ind2)