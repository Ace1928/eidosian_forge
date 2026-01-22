import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
def test_comparison_methods_scalar_pd_na(comparison_op, dtype):
    op_name = f'__{comparison_op.__name__}__'
    a = pd.array(['a', None, 'c'], dtype=dtype)
    result = getattr(a, op_name)(pd.NA)
    if dtype.storage == 'pyarrow_numpy':
        if operator.ne == comparison_op:
            expected = np.array([True, True, True])
        else:
            expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(result, expected)
    else:
        expected_dtype = 'boolean[pyarrow]' if dtype.storage == 'pyarrow' else 'boolean'
        expected = pd.array([None, None, None], dtype=expected_dtype)
        tm.assert_extension_array_equal(result, expected)
        tm.assert_extension_array_equal(result, expected)