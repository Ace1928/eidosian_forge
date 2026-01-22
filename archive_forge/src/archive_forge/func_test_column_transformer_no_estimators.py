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
def test_column_transformer_no_estimators():
    X_array = np.array([[0, 1, 2], [2, 4, 6], [8, 6, 4]]).astype('float').T
    ct = ColumnTransformer([], remainder=StandardScaler())
    params = ct.get_params()
    assert params['remainder__with_mean']
    X_trans = ct.fit_transform(X_array)
    assert X_trans.shape == X_array.shape
    assert len(ct.transformers_) == 1
    assert ct.transformers_[-1][0] == 'remainder'
    assert ct.transformers_[-1][2] == [0, 1, 2]