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
def test_2D_transformer_output():
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    ct = ColumnTransformer([('trans1', 'drop', 0), ('trans2', TransNo2D(), 1)])
    msg = "the 'trans2' transformer should be 2D"
    with pytest.raises(ValueError, match=msg):
        ct.fit_transform(X_array)
    with pytest.raises(ValueError, match=msg):
        ct.fit(X_array)