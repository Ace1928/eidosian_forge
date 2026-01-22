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
def test_column_transformer():
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_res_first1D = np.array([0, 1, 2])
    X_res_second1D = np.array([2, 4, 6])
    X_res_first = X_res_first1D.reshape(-1, 1)
    X_res_both = X_array
    cases = [(0, X_res_first), ([0], X_res_first), ([0, 1], X_res_both), (np.array([0, 1]), X_res_both), (slice(0, 1), X_res_first), (slice(0, 2), X_res_both), (np.array([True, False]), X_res_first), ([True, False], X_res_first), (np.array([True, True]), X_res_both), ([True, True], X_res_both)]
    for selection, res in cases:
        ct = ColumnTransformer([('trans', Trans(), selection)], remainder='drop')
        assert_array_equal(ct.fit_transform(X_array), res)
        assert_array_equal(ct.fit(X_array).transform(X_array), res)
        ct = ColumnTransformer([('trans', Trans(), lambda x: selection)], remainder='drop')
        assert_array_equal(ct.fit_transform(X_array), res)
        assert_array_equal(ct.fit(X_array).transform(X_array), res)
    ct = ColumnTransformer([('trans1', Trans(), [0]), ('trans2', Trans(), [1])])
    assert_array_equal(ct.fit_transform(X_array), X_res_both)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both)
    assert len(ct.transformers_) == 2
    transformer_weights = {'trans1': 0.1, 'trans2': 10}
    both = ColumnTransformer([('trans1', Trans(), [0]), ('trans2', Trans(), [1])], transformer_weights=transformer_weights)
    res = np.vstack([transformer_weights['trans1'] * X_res_first1D, transformer_weights['trans2'] * X_res_second1D]).T
    assert_array_equal(both.fit_transform(X_array), res)
    assert_array_equal(both.fit(X_array).transform(X_array), res)
    assert len(both.transformers_) == 2
    both = ColumnTransformer([('trans', Trans(), [0, 1])], transformer_weights={'trans': 0.1})
    assert_array_equal(both.fit_transform(X_array), 0.1 * X_res_both)
    assert_array_equal(both.fit(X_array).transform(X_array), 0.1 * X_res_both)
    assert len(both.transformers_) == 1