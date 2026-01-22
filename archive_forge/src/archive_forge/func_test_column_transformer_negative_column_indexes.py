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
def test_column_transformer_negative_column_indexes():
    X = np.random.randn(2, 2)
    X_categories = np.array([[1], [2]])
    X = np.concatenate([X, X_categories], axis=1)
    ohe = OneHotEncoder()
    tf_1 = ColumnTransformer([('ohe', ohe, [-1])], remainder='passthrough')
    tf_2 = ColumnTransformer([('ohe', ohe, [2])], remainder='passthrough')
    assert_array_equal(tf_1.fit_transform(X), tf_2.fit_transform(X))