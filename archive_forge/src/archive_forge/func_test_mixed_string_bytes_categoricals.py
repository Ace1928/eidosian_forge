import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_mixed_string_bytes_categoricals():
    """Check that this mixture of predefined categories and X raises an error.

    Categories defined as bytes can not easily be compared to data that is
    a string.
    """
    X = np.array([['b'], ['a']], dtype='U')
    categories = [np.array(['b', 'a'], dtype='S')]
    ohe = OneHotEncoder(categories=categories, sparse_output=False)
    msg = re.escape("In column 0, the predefined categories have type 'bytes' which is incompatible with values of type 'str_'.")
    with pytest.raises(ValueError, match=msg):
        ohe.fit(X)