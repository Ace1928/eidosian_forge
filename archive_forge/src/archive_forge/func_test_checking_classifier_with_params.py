import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sklearn.datasets import load_iris
from sklearn.utils import _safe_indexing, check_array
from sklearn.utils._mocking import (
from sklearn.utils._testing import _convert_container
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_checking_classifier_with_params(iris, csr_container):
    X, y = iris
    X_sparse = csr_container(X)
    clf = CheckingClassifier(check_X=sparse.issparse)
    with pytest.raises(AssertionError):
        clf.fit(X, y)
    clf.fit(X_sparse, y)
    clf = CheckingClassifier(check_X=check_array, check_X_params={'accept_sparse': False})
    clf.fit(X, y)
    with pytest.raises(TypeError, match='Sparse data was passed'):
        clf.fit(X_sparse, y)