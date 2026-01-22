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
def test_column_transformer_tuple_transformers_parameter():
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    transformers = [('trans1', Trans(), [0]), ('trans2', Trans(), [1])]
    ct_with_list = ColumnTransformer(transformers)
    ct_with_tuple = ColumnTransformer(tuple(transformers))
    assert_array_equal(ct_with_list.fit_transform(X_array), ct_with_tuple.fit_transform(X_array))
    assert_array_equal(ct_with_list.fit(X_array).transform(X_array), ct_with_tuple.fit(X_array).transform(X_array))