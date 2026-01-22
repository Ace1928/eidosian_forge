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
def test_make_column_selector_error():
    selector = make_column_selector(dtype_include=np.number)
    X = np.array([[0.1, 0.2]])
    msg = 'make_column_selector can only be applied to pandas dataframes'
    with pytest.raises(ValueError, match=msg):
        selector(X)