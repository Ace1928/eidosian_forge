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
@pytest.mark.parametrize('NeighborsHeap', [NeighborsHeapBT, NeighborsHeapKDT])
def test_neighbors_heap(NeighborsHeap, n_pts=5, n_nbrs=10):
    heap = NeighborsHeap(n_pts, n_nbrs)
    rng = check_random_state(0)
    for row in range(n_pts):
        d_in = rng.random_sample(2 * n_nbrs).astype(np.float64, copy=False)
        i_in = np.arange(2 * n_nbrs, dtype=np.intp)
        for d, i in zip(d_in, i_in):
            heap.push(row, d, i)
        ind = np.argsort(d_in)
        d_in = d_in[ind]
        i_in = i_in[ind]
        d_heap, i_heap = heap.get_arrays(sort=True)
        assert_array_almost_equal(d_in[:n_nbrs], d_heap[row])
        assert_array_almost_equal(i_in[:n_nbrs], i_heap[row])