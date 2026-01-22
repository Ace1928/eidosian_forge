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
@pytest.mark.parametrize('constructor_name', ['dataframe', 'polars'])
def test_column_transformer_dataframe(constructor_name):
    if constructor_name == 'dataframe':
        dataframe_lib = pytest.importorskip('pandas')
    else:
        dataframe_lib = pytest.importorskip(constructor_name)
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_df = _convert_container(X_array, constructor_name, columns_name=['first', 'second'])
    X_res_first = np.array([0, 1, 2]).reshape(-1, 1)
    X_res_both = X_array
    cases = [(['first'], X_res_first), (['first', 'second'], X_res_both), (slice('first', 'second'), X_res_both), ([0], X_res_first), ([0, 1], X_res_both), (np.array([0, 1]), X_res_both), (slice(0, 1), X_res_first), (slice(0, 2), X_res_both), (np.array([True, False]), X_res_first), ([True, False], X_res_first)]
    if constructor_name == 'dataframe':
        cases.extend([(0, X_res_first), ('first', X_res_first), (dataframe_lib.Series([True, False], index=['first', 'second']), X_res_first)])
    for selection, res in cases:
        ct = ColumnTransformer([('trans', Trans(), selection)], remainder='drop')
        assert_array_equal(ct.fit_transform(X_df), res)
        assert_array_equal(ct.fit(X_df).transform(X_df), res)
        ct = ColumnTransformer([('trans', Trans(), lambda X: selection)], remainder='drop')
        assert_array_equal(ct.fit_transform(X_df), res)
        assert_array_equal(ct.fit(X_df).transform(X_df), res)
    ct = ColumnTransformer([('trans1', Trans(), ['first']), ('trans2', Trans(), ['second'])])
    assert_array_equal(ct.fit_transform(X_df), X_res_both)
    assert_array_equal(ct.fit(X_df).transform(X_df), X_res_both)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] != 'remainder'
    ct = ColumnTransformer([('trans1', Trans(), [0]), ('trans2', Trans(), [1])])
    assert_array_equal(ct.fit_transform(X_df), X_res_both)
    assert_array_equal(ct.fit(X_df).transform(X_df), X_res_both)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] != 'remainder'
    transformer_weights = {'trans1': 0.1, 'trans2': 10}
    both = ColumnTransformer([('trans1', Trans(), ['first']), ('trans2', Trans(), ['second'])], transformer_weights=transformer_weights)
    res = np.vstack([transformer_weights['trans1'] * X_df['first'], transformer_weights['trans2'] * X_df['second']]).T
    assert_array_equal(both.fit_transform(X_df), res)
    assert_array_equal(both.fit(X_df).transform(X_df), res)
    assert len(both.transformers_) == 2
    assert both.transformers_[-1][0] != 'remainder'
    both = ColumnTransformer([('trans', Trans(), ['first', 'second'])], transformer_weights={'trans': 0.1})
    assert_array_equal(both.fit_transform(X_df), 0.1 * X_res_both)
    assert_array_equal(both.fit(X_df).transform(X_df), 0.1 * X_res_both)
    assert len(both.transformers_) == 1
    assert both.transformers_[-1][0] != 'remainder'
    both = ColumnTransformer([('trans', Trans(), [0, 1])], transformer_weights={'trans': 0.1})
    assert_array_equal(both.fit_transform(X_df), 0.1 * X_res_both)
    assert_array_equal(both.fit(X_df).transform(X_df), 0.1 * X_res_both)
    assert len(both.transformers_) == 1
    assert both.transformers_[-1][0] != 'remainder'

    class TransAssert(BaseEstimator):

        def __init__(self, expected_type_transform):
            self.expected_type_transform = expected_type_transform

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            assert isinstance(X, self.expected_type_transform)
            if isinstance(X, dataframe_lib.Series):
                X = X.to_frame()
            return X
    ct = ColumnTransformer([('trans', TransAssert(expected_type_transform=dataframe_lib.DataFrame), ['first', 'second'])])
    ct.fit_transform(X_df)
    if constructor_name == 'dataframe':
        ct = ColumnTransformer([('trans', TransAssert(expected_type_transform=dataframe_lib.Series), 'first')], remainder='drop')
        ct.fit_transform(X_df)
        X_df2 = X_df.copy()
        X_df2.columns = [1, 0]
        ct = ColumnTransformer([('trans', Trans(), 0)], remainder='drop')
        assert_array_equal(ct.fit_transform(X_df2), X_res_first)
        assert_array_equal(ct.fit(X_df2).transform(X_df2), X_res_first)
        assert len(ct.transformers_) == 2
        assert ct.transformers_[-1][0] == 'remainder'
        assert ct.transformers_[-1][1] == 'drop'
        assert_array_equal(ct.transformers_[-1][2], [1])