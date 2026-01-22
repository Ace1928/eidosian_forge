import re
import warnings
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
@pytest.mark.parametrize('original, dtype, expected_error_msg', [(SparseDtype(float, np.nan), int, re.escape('Cannot convert non-finite values (NA or inf) to integer')), (SparseDtype(str, 'abc'), int, "invalid literal for int\\(\\) with base 10: ('abc'|np\\.str_\\('abc'\\))")])
def test_update_dtype_raises(original, dtype, expected_error_msg):
    with pytest.raises(ValueError, match=expected_error_msg):
        original.update_dtype(dtype)