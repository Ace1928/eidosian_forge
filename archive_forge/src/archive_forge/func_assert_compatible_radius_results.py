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
def assert_compatible_radius_results(neighbors_dists_a, neighbors_dists_b, neighbors_indices_a, neighbors_indices_b, radius, check_sorted=True, rtol=1e-05, atol=1e-06):
    """Assert that radius neighborhood results are valid up to:

      - relative and absolute tolerance on computed distance values
      - permutations of indices for distances values that differ up to
        a precision level
      - missing or extra last elements if their distance is
        close to the radius

    To be used for testing neighbors queries on float32 datasets: we
    accept neighbors rank swaps only if they are caused by small
    rounding errors on the distance computations.

    Input arrays must be sorted w.r.t distances.
    """
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    assert len(neighbors_dists_a) == len(neighbors_dists_b) == len(neighbors_indices_a) == len(neighbors_indices_b)
    n_queries = len(neighbors_dists_a)
    for query_idx in range(n_queries):
        dist_row_a = neighbors_dists_a[query_idx]
        dist_row_b = neighbors_dists_b[query_idx]
        indices_row_a = neighbors_indices_a[query_idx]
        indices_row_b = neighbors_indices_b[query_idx]
        if check_sorted:
            assert is_sorted(dist_row_a), f"Distances aren't sorted on row {query_idx}"
            assert is_sorted(dist_row_b), f"Distances aren't sorted on row {query_idx}"
        assert len(dist_row_a) == len(indices_row_a)
        assert len(dist_row_b) == len(indices_row_b)
        if len(dist_row_a) > 0:
            max_dist_a = np.max(dist_row_a)
            assert max_dist_a <= radius, f'Largest returned distance {max_dist_a} not within requested radius {radius} on row {query_idx}'
        if len(dist_row_b) > 0:
            max_dist_b = np.max(dist_row_b)
            assert max_dist_b <= radius, f'Largest returned distance {max_dist_b} not within requested radius {radius} on row {query_idx}'
        assert_same_distances_for_common_neighbors(query_idx, dist_row_a, dist_row_b, indices_row_a, indices_row_b, rtol, atol)
        threshold = (1 - rtol) * radius - atol
        assert_no_missing_neighbors(query_idx, dist_row_a, dist_row_b, indices_row_a, indices_row_b, threshold)