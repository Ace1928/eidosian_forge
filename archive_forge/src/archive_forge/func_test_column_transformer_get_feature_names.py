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
def test_column_transformer_get_feature_names():
    X_array = np.array([[0.0, 1.0, 2.0], [2.0, 4.0, 6.0]]).T
    ct = ColumnTransformer([('trans', Trans(), [0, 1])])
    with pytest.raises(NotFittedError):
        ct.get_feature_names_out()
    ct.fit(X_array)
    msg = re.escape('Transformer trans (type Trans) does not provide get_feature_names_out')
    with pytest.raises(AttributeError, match=msg):
        ct.get_feature_names_out()