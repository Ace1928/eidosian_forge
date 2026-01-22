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
@pytest.mark.parametrize('key', [[0], slice(0, 1), np.array([True, False]), ['first'], 'pd-index', np.array(['first']), np.array(['first'], dtype=object), slice(None, 'first'), slice('first', 'first')])
def test_column_transformer_remainder_pandas(key):
    pd = pytest.importorskip('pandas')
    if isinstance(key, str) and key == 'pd-index':
        key = pd.Index(['first'])
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_df = pd.DataFrame(X_array, columns=['first', 'second'])
    X_res_both = X_array
    ct = ColumnTransformer([('trans1', Trans(), key)], remainder='passthrough')
    assert_array_equal(ct.fit_transform(X_df), X_res_both)
    assert_array_equal(ct.fit(X_df).transform(X_df), X_res_both)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == 'remainder'
    assert isinstance(ct.transformers_[-1][1], FunctionTransformer)
    assert_array_equal(ct.transformers_[-1][2], [1])