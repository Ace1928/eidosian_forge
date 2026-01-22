import itertools
import re
import warnings
from functools import partial
import numpy as np
import pytest
import threadpoolctl
from scipy.spatial.distance import cdist
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.metrics._pairwise_distances_reduction import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('check_sorted', [True, False])
def test_assert_compatible_radius_results(check_sorted):
    atol = 1e-07
    rtol = 0.0
    tols = dict(atol=atol, rtol=rtol)
    eps = atol / 3
    _1m = 1.0 - eps
    _1p = 1.0 + eps
    _6_1m = 6.1 - eps
    _6_1p = 6.1 + eps
    ref_dist = [np.array([1.2, 2.5, _6_1m, 6.1, _6_1p]), np.array([_1m, 1, _1p, _1p])]
    ref_indices = [np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8, 9])]
    assert_compatible_radius_results(ref_dist, ref_dist, ref_indices, ref_indices, radius=7.0, check_sorted=check_sorted, **tols)
    assert_compatible_radius_results(np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]), np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]), np.array([np.array([1, 2, 3, 4, 5])]), np.array([np.array([1, 2, 4, 5, 3])]), radius=7.0, check_sorted=check_sorted, **tols)
    assert_compatible_radius_results(np.array([np.array([_1m, _1m, 1, _1p, _1p])]), np.array([np.array([_1m, _1m, 1, _1p, _1p])]), np.array([np.array([6, 7, 8, 9, 10])]), np.array([np.array([6, 9, 7, 8, 10])]), radius=7.0, check_sorted=check_sorted, **tols)
    msg = re.escape('Query vector with index 0 lead to different distances for common neighbor with index 1: dist_a=1.2 vs dist_b=2.5')
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_radius_results(np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]), np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]), np.array([np.array([1, 2, 3, 4, 5])]), np.array([np.array([2, 1, 3, 4, 5])]), radius=7.0, check_sorted=check_sorted, **tols)
    assert_compatible_radius_results(np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p, _6_1p])]), np.array([np.array([1.2, 2.5, _6_1m, 6.1])]), np.array([np.array([1, 2, 3, 4, 5, 7])]), np.array([np.array([1, 2, 3, 6])]), radius=_6_1p, check_sorted=check_sorted, **tols)
    msg = re.escape('Query vector with index 0 lead to mismatched result indices:\nneighbors in b missing from a: []\nneighbors in a missing from b: [3]')
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_radius_results(np.array([np.array([1.2, 2.5, 6])]), np.array([np.array([1.2, 2.5])]), np.array([np.array([1, 2, 3])]), np.array([np.array([1, 2])]), radius=6.1, check_sorted=check_sorted, **tols)
    msg = re.escape('Query vector with index 0 lead to mismatched result indices:\nneighbors in b missing from a: [4]\nneighbors in a missing from b: [2]')
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_radius_results(np.array([np.array([1.2, 2.1, 2.5])]), np.array([np.array([1.2, 2, 2.5])]), np.array([np.array([1, 2, 3])]), np.array([np.array([1, 4, 3])]), radius=6.1, check_sorted=check_sorted, **tols)
    msg = re.escape('Largest returned distance 6.100000033333333 not within requested radius 6.1 on row 0')
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_radius_results(np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]), np.array([np.array([1.2, 2.5, _6_1m, 6.1, 6.1])]), np.array([np.array([1, 2, 3, 4, 5])]), np.array([np.array([2, 1, 4, 5, 3])]), radius=6.1, check_sorted=check_sorted, **tols)
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_radius_results(np.array([np.array([1.2, 2.5, _6_1m, 6.1, 6.1])]), np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]), np.array([np.array([1, 2, 3, 4, 5])]), np.array([np.array([2, 1, 4, 5, 3])]), radius=6.1, check_sorted=check_sorted, **tols)
    if check_sorted:
        msg = "Distances aren't sorted on row 0"
        with pytest.raises(AssertionError, match=msg):
            assert_compatible_radius_results(np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]), np.array([np.array([2.5, 1.2, _6_1m, 6.1, _6_1p])]), np.array([np.array([1, 2, 3, 4, 5])]), np.array([np.array([2, 1, 4, 5, 3])]), radius=_6_1p, check_sorted=True, **tols)
    else:
        assert_compatible_radius_results(np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]), np.array([np.array([2.5, 1.2, _6_1m, 6.1, _6_1p])]), np.array([np.array([1, 2, 3, 4, 5])]), np.array([np.array([2, 1, 4, 5, 3])]), radius=_6_1p, check_sorted=False, **tols)