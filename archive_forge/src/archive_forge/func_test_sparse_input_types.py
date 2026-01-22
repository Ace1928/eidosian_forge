import warnings
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.semi_supervised import _label_propagation as label_propagation
from sklearn.utils._testing import (
@pytest.mark.parametrize('accepted_sparse_type', ['sparse_csr', 'sparse_csc'])
@pytest.mark.parametrize('index_dtype', [np.int32, np.int64])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('Estimator, parameters', ESTIMATORS)
def test_sparse_input_types(accepted_sparse_type, index_dtype, dtype, Estimator, parameters):
    X = _convert_container([[1.0, 0.0], [0.0, 2.0], [1.0, 3.0]], accepted_sparse_type)
    X.data = X.data.astype(dtype, copy=False)
    X.indices = X.indices.astype(index_dtype, copy=False)
    X.indptr = X.indptr.astype(index_dtype, copy=False)
    labels = [0, 1, -1]
    clf = Estimator(**parameters).fit(X, labels)
    assert_array_equal(clf.predict([[0.5, 2.5]]), np.array([1]))