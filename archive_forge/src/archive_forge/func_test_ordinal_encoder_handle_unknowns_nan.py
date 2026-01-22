import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ordinal_encoder_handle_unknowns_nan():
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    X_fit = np.array([[1], [2], [3]])
    enc.fit(X_fit)
    X_trans = enc.transform([[1], [2], [4]])
    assert_array_equal(X_trans, [[0], [1], [np.nan]])