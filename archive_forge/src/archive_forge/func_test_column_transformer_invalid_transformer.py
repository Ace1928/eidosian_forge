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
def test_column_transformer_invalid_transformer():

    class NoTrans(BaseEstimator):

        def fit(self, X, y=None):
            return self

        def predict(self, X):
            return X
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    ct = ColumnTransformer([('trans', NoTrans(), [0])])
    msg = 'All estimators should implement fit and transform'
    with pytest.raises(TypeError, match=msg):
        ct.fit(X_array)