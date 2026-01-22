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
@pytest.mark.parametrize('remainder', ['drop', 'passthrough'])
def test_column_transformer_invalid_columns(remainder):
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    for col in [1.5, ['string', 1], slice(1, 's'), np.array([1.0])]:
        ct = ColumnTransformer([('trans', Trans(), col)], remainder=remainder)
        with pytest.raises(ValueError, match='No valid specification'):
            ct.fit(X_array)
    for col in ['string', ['string', 'other'], slice('a', 'b')]:
        ct = ColumnTransformer([('trans', Trans(), col)], remainder=remainder)
        with pytest.raises(ValueError, match='Specifying the columns'):
            ct.fit(X_array)
    col = [0, 1]
    ct = ColumnTransformer([('trans', Trans(), col)], remainder=remainder)
    ct.fit(X_array)
    X_array_more = np.array([[0, 1, 2], [2, 4, 6], [3, 6, 9]]).T
    msg = 'X has 3 features, but ColumnTransformer is expecting 2 features as input.'
    with pytest.raises(ValueError, match=msg):
        ct.transform(X_array_more)
    X_array_fewer = np.array([[0, 1, 2]]).T
    err_msg = 'X has 1 features, but ColumnTransformer is expecting 2 features as input.'
    with pytest.raises(ValueError, match=err_msg):
        ct.transform(X_array_fewer)