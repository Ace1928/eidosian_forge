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
def test_column_transformer_named_estimators():
    X_array = np.array([[0.0, 1.0, 2.0], [2.0, 4.0, 6.0]]).T
    ct = ColumnTransformer([('trans1', StandardScaler(), [0]), ('trans2', StandardScaler(with_std=False), [1])])
    assert not hasattr(ct, 'transformers_')
    ct.fit(X_array)
    assert hasattr(ct, 'transformers_')
    assert isinstance(ct.named_transformers_['trans1'], StandardScaler)
    assert isinstance(ct.named_transformers_.trans1, StandardScaler)
    assert isinstance(ct.named_transformers_['trans2'], StandardScaler)
    assert isinstance(ct.named_transformers_.trans2, StandardScaler)
    assert not ct.named_transformers_.trans2.with_std
    assert ct.named_transformers_.trans1.mean_ == 1.0