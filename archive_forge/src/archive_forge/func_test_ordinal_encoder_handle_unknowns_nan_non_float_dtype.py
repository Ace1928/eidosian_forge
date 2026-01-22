import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ordinal_encoder_handle_unknowns_nan_non_float_dtype():
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, dtype=int)
    X_fit = np.array([[1], [2], [3]])
    with pytest.raises(ValueError, match='dtype parameter should be a float dtype'):
        enc.fit(X_fit)