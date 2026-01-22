import pickle
import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import (
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
from sklearn.tests.metadata_routing_common import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_column_transformer_sparse_remainder_transformer(csr_container):
    X_array = np.array([[0, 1, 2], [2, 4, 6], [8, 6, 4]]).T
    ct = ColumnTransformer([('trans1', Trans(), [0])], remainder=SparseMatrixTrans(csr_container), sparse_threshold=0.8)
    X_trans = ct.fit_transform(X_array)
    assert sparse.issparse(X_trans)
    assert X_trans.shape == (3, 3 + 1)
    exp_array = np.hstack((X_array[:, 0].reshape(-1, 1), np.eye(3)))
    assert_array_equal(X_trans.toarray(), exp_array)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == 'remainder'
    assert isinstance(ct.transformers_[-1][1], SparseMatrixTrans)
    assert_array_equal(ct.transformers_[-1][2], [1, 2])