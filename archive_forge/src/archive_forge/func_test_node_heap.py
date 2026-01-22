import itertools
import pickle
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from sklearn.metrics import DistanceMetric
from sklearn.neighbors._ball_tree import (
from sklearn.neighbors._ball_tree import (
from sklearn.neighbors._ball_tree import (
from sklearn.neighbors._ball_tree import (
from sklearn.neighbors._kd_tree import (
from sklearn.neighbors._kd_tree import (
from sklearn.neighbors._kd_tree import (
from sklearn.neighbors._kd_tree import (
from sklearn.utils import check_random_state
@pytest.mark.parametrize('nodeheap_sort', [nodeheap_sort_bt, nodeheap_sort_kdt])
def test_node_heap(nodeheap_sort, n_nodes=50):
    rng = check_random_state(0)
    vals = rng.random_sample(n_nodes).astype(np.float64, copy=False)
    i1 = np.argsort(vals)
    vals2, i2 = nodeheap_sort(vals)
    assert_array_almost_equal(i1, i2)
    assert_array_almost_equal(vals[i1], vals2)