import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
def test_contains_with_float_index(self, any_real_numpy_dtype):
    dtype = any_real_numpy_dtype
    data = [0, 1, 2, 3] if not is_float_dtype(dtype) else [0.1, 1.1, 2.2, 3.3]
    index = Index(data, dtype=dtype)
    if not is_float_dtype(index.dtype):
        assert 1.1 not in index
        assert 1.0 in index
        assert 1 in index
    else:
        assert 1.1 in index
        assert 1.0 not in index
        assert 1 not in index