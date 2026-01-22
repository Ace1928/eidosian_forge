import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ordinal_encoder_missing_appears_frequent():
    """Check behavior when missing value appears frequently."""
    X = np.array([[np.nan] * 20 + ['dog'] * 10 + ['cat'] * 5 + ['snake'] + ['deer']], dtype=object).T
    ordinal = OrdinalEncoder(max_categories=3).fit(X)
    X_test = np.array([['snake', 'cat', 'dog', np.nan]], dtype=object).T
    X_trans = ordinal.transform(X_test)
    assert_allclose(X_trans, [[2], [0], [1], [np.nan]])