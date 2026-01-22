import itertools
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_equal
from sklearn.neighbors._ball_tree import BallTree, BallTree32, BallTree64
from sklearn.utils import check_random_state
from sklearn.utils._testing import _convert_container
from sklearn.utils.validation import check_array
@pytest.mark.parametrize('metric', itertools.chain(BOOLEAN_METRICS, DISCRETE_METRICS))
@pytest.mark.parametrize('array_type', ['list', 'array'])
@pytest.mark.parametrize('BallTreeImplementation', BALL_TREE_CLASSES)
def test_ball_tree_query_metrics(metric, array_type, BallTreeImplementation):
    rng = check_random_state(0)
    if metric in BOOLEAN_METRICS:
        X = rng.random_sample((40, 10)).round(0)
        Y = rng.random_sample((10, 10)).round(0)
    elif metric in DISCRETE_METRICS:
        X = (4 * rng.random_sample((40, 10))).round(0)
        Y = (4 * rng.random_sample((10, 10))).round(0)
    X = _convert_container(X, array_type)
    Y = _convert_container(Y, array_type)
    k = 5
    bt = BallTreeImplementation(X, leaf_size=1, metric=metric)
    dist1, ind1 = bt.query(Y, k)
    dist2, ind2 = brute_force_neighbors(X, Y, k, metric)
    assert_array_almost_equal(dist1, dist2)