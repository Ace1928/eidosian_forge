import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ohe_infrequent_mixed():
    """Test infrequent categories where feature 0 has infrequent categories,
    and feature 1 does not."""
    X = np.c_[[0, 1, 3, 3, 3, 3, 2, 0, 3], [0, 0, 0, 0, 1, 1, 1, 1, 1]]
    ohe = OneHotEncoder(max_categories=3, drop='if_binary', sparse_output=False)
    ohe.fit(X)
    X_test = [[3, 0], [1, 1]]
    X_trans = ohe.transform(X_test)
    assert_allclose(X_trans, [[0, 1, 0, 0], [0, 0, 1, 1]])