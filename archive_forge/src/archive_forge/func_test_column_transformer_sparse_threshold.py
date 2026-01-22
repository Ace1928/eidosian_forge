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
def test_column_transformer_sparse_threshold():
    X_array = np.array([['a', 'b'], ['A', 'B']], dtype=object).T
    col_trans = ColumnTransformer([('trans1', OneHotEncoder(), [0]), ('trans2', OneHotEncoder(), [1])], sparse_threshold=0.2)
    res = col_trans.fit_transform(X_array)
    assert not sparse.issparse(res)
    assert not col_trans.sparse_output_
    for thres in [0.75001, 1]:
        col_trans = ColumnTransformer([('trans1', OneHotEncoder(sparse_output=True), [0]), ('trans2', OneHotEncoder(sparse_output=False), [1])], sparse_threshold=thres)
        res = col_trans.fit_transform(X_array)
        assert sparse.issparse(res)
        assert col_trans.sparse_output_
    for thres in [0.75, 0]:
        col_trans = ColumnTransformer([('trans1', OneHotEncoder(sparse_output=True), [0]), ('trans2', OneHotEncoder(sparse_output=False), [1])], sparse_threshold=thres)
        res = col_trans.fit_transform(X_array)
        assert not sparse.issparse(res)
        assert not col_trans.sparse_output_
    for thres in [0.33, 0, 1]:
        col_trans = ColumnTransformer([('trans1', OneHotEncoder(sparse_output=False), [0]), ('trans2', OneHotEncoder(sparse_output=False), [1])], sparse_threshold=thres)
        res = col_trans.fit_transform(X_array)
        assert not sparse.issparse(res)
        assert not col_trans.sparse_output_