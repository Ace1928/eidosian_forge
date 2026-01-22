import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ordinal_encoder_passthrough_missing_values_float_errors_dtype():
    """Test ordinal encoder with nan passthrough fails when dtype=np.int32."""
    X = np.array([[np.nan, 3.0, 1.0, 3.0]]).T
    oe = OrdinalEncoder(dtype=np.int32)
    msg = f'There are missing values in features \\[0\\]. For OrdinalEncoder to encode missing values with dtype: {np.int32}'
    with pytest.raises(ValueError, match=msg):
        oe.fit(X)