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
def assert_no_missing_neighbors(query_idx, dist_row_a, dist_row_b, indices_row_a, indices_row_b, threshold):
    """Compare the indices of neighbors in two results sets.

    Any neighbor index with a distance below the precision threshold should
    match one in the other result set. We ignore the last few neighbors beyond
    the threshold as those can typically be missing due to rounding errors.

    For radius queries, the threshold is just the radius minus the expected
    precision level.

    For k-NN queries, it is the maximum distance to the k-th neighbor minus the
    expected precision level.
    """
    mask_a = dist_row_a < threshold
    mask_b = dist_row_b < threshold
    missing_from_b = np.setdiff1d(indices_row_a[mask_a], indices_row_b)
    missing_from_a = np.setdiff1d(indices_row_b[mask_b], indices_row_a)
    if len(missing_from_a) > 0 or len(missing_from_b) > 0:
        raise AssertionError(f'Query vector with index {query_idx} lead to mismatched result indices:\nneighbors in b missing from a: {missing_from_a}\nneighbors in a missing from b: {missing_from_b}\ndist_row_a={dist_row_a}\ndist_row_b={dist_row_b}\nindices_row_a={indices_row_a}\nindices_row_b={indices_row_b}\n')