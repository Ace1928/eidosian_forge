import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('drop', [['abc', 3], ['abc', 3, 41, 'a']])
def test_invalid_drop_length(drop):
    enc = OneHotEncoder(drop=drop)
    err_msg = '`drop` should have length equal to the number'
    with pytest.raises(ValueError, match=err_msg):
        enc.fit([['abc', 2, 55], ['def', 1, 55], ['def', 3, 59]])