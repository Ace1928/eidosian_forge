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
def test_column_transformer_output_indices():
    X_array = np.arange(6).reshape(3, 2)
    ct = ColumnTransformer([('trans1', Trans(), [0]), ('trans2', Trans(), [1])])
    X_trans = ct.fit_transform(X_array)
    assert ct.output_indices_ == {'trans1': slice(0, 1), 'trans2': slice(1, 2), 'remainder': slice(0, 0)}
    assert_array_equal(X_trans[:, [0]], X_trans[:, ct.output_indices_['trans1']])
    assert_array_equal(X_trans[:, [1]], X_trans[:, ct.output_indices_['trans2']])
    ct = ColumnTransformer([('trans', Trans(), [0, 1])], transformer_weights={'trans': 0.1})
    X_trans = ct.fit_transform(X_array)
    assert ct.output_indices_ == {'trans': slice(0, 2), 'remainder': slice(0, 0)}
    assert_array_equal(X_trans[:, [0, 1]], X_trans[:, ct.output_indices_['trans']])
    assert_array_equal(X_trans[:, []], X_trans[:, ct.output_indices_['remainder']])
    ct = ColumnTransformer([('trans1', Trans(), [0, 1]), ('trans2', TransRaise(), [])])
    X_trans = ct.fit_transform(X_array)
    assert ct.output_indices_ == {'trans1': slice(0, 2), 'trans2': slice(0, 0), 'remainder': slice(0, 0)}
    assert_array_equal(X_trans[:, [0, 1]], X_trans[:, ct.output_indices_['trans1']])
    assert_array_equal(X_trans[:, []], X_trans[:, ct.output_indices_['trans2']])
    assert_array_equal(X_trans[:, []], X_trans[:, ct.output_indices_['remainder']])
    ct = ColumnTransformer([('trans', TransRaise(), [])], remainder='passthrough')
    X_trans = ct.fit_transform(X_array)
    assert ct.output_indices_ == {'trans': slice(0, 0), 'remainder': slice(0, 2)}
    assert_array_equal(X_trans[:, []], X_trans[:, ct.output_indices_['trans']])
    assert_array_equal(X_trans[:, [0, 1]], X_trans[:, ct.output_indices_['remainder']])