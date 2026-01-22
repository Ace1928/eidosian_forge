import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from sklearn.neighbors._kd_tree import KDTree, KDTree32, KDTree64
from sklearn.neighbors.tests.test_ball_tree import get_dataset_for_binary_tree
from sklearn.utils.parallel import Parallel, delayed
@pytest.mark.parametrize('metric', METRICS)
def test_kd_tree_numerical_consistency(global_random_seed, metric):
    X_64, X_32, Y_64, Y_32 = get_dataset_for_binary_tree(random_seed=global_random_seed, features=50)
    metric_params = METRICS.get(metric, {})
    kd_64 = KDTree64(X_64, leaf_size=2, metric=metric, **metric_params)
    kd_32 = KDTree32(X_32, leaf_size=2, metric=metric, **metric_params)
    k = 4
    dist_64, ind_64 = kd_64.query(Y_64, k=k)
    dist_32, ind_32 = kd_32.query(Y_32, k=k)
    assert_allclose(dist_64, dist_32, rtol=1e-05)
    assert_equal(ind_64, ind_32)
    assert dist_64.dtype == np.float64
    assert dist_32.dtype == np.float32
    r = 2.38
    ind_64 = kd_64.query_radius(Y_64, r=r)
    ind_32 = kd_32.query_radius(Y_32, r=r)
    for _ind64, _ind32 in zip(ind_64, ind_32):
        assert_equal(_ind64, _ind32)
    ind_64, dist_64 = kd_64.query_radius(Y_64, r=r, return_distance=True)
    ind_32, dist_32 = kd_32.query_radius(Y_32, r=r, return_distance=True)
    for _ind64, _ind32, _dist_64, _dist_32 in zip(ind_64, ind_32, dist_64, dist_32):
        assert_equal(_ind64, _ind32)
        assert_allclose(_dist_64, _dist_32, rtol=1e-05)
        assert _dist_64.dtype == np.float64
        assert _dist_32.dtype == np.float32