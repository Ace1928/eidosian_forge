import sys
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.sparse import random as sparse_random
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._csr_polynomial_expansion import (
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils.fixes import (
@pytest.mark.parametrize('interaction_only', [True, False])
@pytest.mark.parametrize('include_bias', [True, False])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_csr_polynomial_expansion_index_overflow_non_regression(interaction_only, include_bias, csr_container):
    """Check the automatic index dtype promotion to `np.int64` when needed.

    This ensures that sufficiently large input configurations get
    properly promoted to use `np.int64` for index and indptr representation
    while preserving data integrity. Non-regression test for gh-16803.

    Note that this is only possible for Python runtimes with a 64 bit address
    space. On 32 bit platforms, a `ValueError` is raised instead.
    """

    def degree_2_calc(d, i, j):
        if interaction_only:
            return d * i - (i ** 2 + 3 * i) // 2 - 1 + j
        else:
            return d * i - (i ** 2 + i) // 2 + j
    n_samples = 13
    n_features = 120001
    data_dtype = np.float32
    data = np.arange(1, 5, dtype=np.int64)
    row = np.array([n_samples - 2, n_samples - 2, n_samples - 1, n_samples - 1])
    col = np.array([n_features - 2, n_features - 1, n_features - 2, n_features - 1], dtype=np.int64)
    X = csr_container((data, (row, col)), shape=(n_samples, n_features), dtype=data_dtype)
    pf = PolynomialFeatures(interaction_only=interaction_only, include_bias=include_bias, degree=2)
    num_combinations = pf._num_combinations(n_features=n_features, min_degree=0, max_degree=2, interaction_only=pf.interaction_only, include_bias=pf.include_bias)
    if num_combinations > np.iinfo(np.intp).max:
        msg = 'The output that would result from the current configuration would have \\d* features which is too large to be indexed'
        with pytest.raises(ValueError, match=msg):
            pf.fit(X)
        return
    X_trans = pf.fit_transform(X)
    row_nonzero, col_nonzero = X_trans.nonzero()
    n_degree_1_features_out = n_features + include_bias
    max_degree_2_idx = degree_2_calc(n_features, col[int(not interaction_only)], col[1]) + n_degree_1_features_out
    data_target = [1] * (n_samples - 2) if include_bias else []
    col_nonzero_target = [0] * (n_samples - 2) if include_bias else []
    for i in range(2):
        x = data[2 * i]
        y = data[2 * i + 1]
        x_idx = col[2 * i]
        y_idx = col[2 * i + 1]
        if include_bias:
            data_target.append(1)
            col_nonzero_target.append(0)
        data_target.extend([x, y])
        col_nonzero_target.extend([x_idx + int(include_bias), y_idx + int(include_bias)])
        if not interaction_only:
            data_target.extend([x * x, x * y, y * y])
            col_nonzero_target.extend([degree_2_calc(n_features, x_idx, x_idx) + n_degree_1_features_out, degree_2_calc(n_features, x_idx, y_idx) + n_degree_1_features_out, degree_2_calc(n_features, y_idx, y_idx) + n_degree_1_features_out])
        else:
            data_target.extend([x * y])
            col_nonzero_target.append(degree_2_calc(n_features, x_idx, y_idx) + n_degree_1_features_out)
    nnz_per_row = int(include_bias) + 3 + 2 * int(not interaction_only)
    assert pf.n_output_features_ == max_degree_2_idx + 1
    assert X_trans.dtype == data_dtype
    assert X_trans.shape == (n_samples, max_degree_2_idx + 1)
    assert X_trans.indptr.dtype == X_trans.indices.dtype == np.int64
    assert X_trans.indices.max() > np.iinfo(np.int32).max
    row_nonzero_target = list(range(n_samples - 2)) if include_bias else []
    row_nonzero_target.extend([n_samples - 2] * nnz_per_row + [n_samples - 1] * nnz_per_row)
    assert_allclose(X_trans.data, data_target)
    assert_array_equal(row_nonzero, row_nonzero_target)
    assert_array_equal(col_nonzero, col_nonzero_target)