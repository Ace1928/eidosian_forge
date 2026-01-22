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
@pytest.mark.parametrize('key', [[0], np.array([0]), slice(0, 1), np.array([True, False, False])])
def test_column_transformer_remainder_transformer(key):
    X_array = np.array([[0, 1, 2], [2, 4, 6], [8, 6, 4]]).T
    X_res_both = X_array.copy()
    X_res_both[:, 1:3] *= 2
    ct = ColumnTransformer([('trans1', Trans(), key)], remainder=DoubleTrans())
    assert_array_equal(ct.fit_transform(X_array), X_res_both)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == 'remainder'
    assert isinstance(ct.transformers_[-1][1], DoubleTrans)
    assert_array_equal(ct.transformers_[-1][2], [1, 2])