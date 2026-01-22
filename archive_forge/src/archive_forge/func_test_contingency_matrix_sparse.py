import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings
from sklearn.utils import assert_all_finite
from sklearn.utils._testing import assert_almost_equal
def test_contingency_matrix_sparse():
    labels_a = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    labels_b = np.array([1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 1, 3, 3, 3, 2, 2])
    C = contingency_matrix(labels_a, labels_b)
    C_sparse = contingency_matrix(labels_a, labels_b, sparse=True).toarray()
    assert_array_almost_equal(C, C_sparse)
    with pytest.raises(ValueError, match="Cannot set 'eps' when sparse=True"):
        contingency_matrix(labels_a, labels_b, eps=1e-10, sparse=True)