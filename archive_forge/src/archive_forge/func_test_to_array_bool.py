import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
from pandas.core.arrays.floating import (
@pytest.mark.parametrize('bool_values, values, target_dtype, expected_dtype', [([False, True], [0, 1], Float64Dtype(), Float64Dtype()), ([False, True], [0, 1], 'Float64', Float64Dtype()), ([False, True, np.nan], [0, 1, np.nan], Float64Dtype(), Float64Dtype())])
def test_to_array_bool(bool_values, values, target_dtype, expected_dtype):
    result = pd.array(bool_values, dtype=target_dtype)
    assert result.dtype == expected_dtype
    expected = pd.array(values, dtype=target_dtype)
    tm.assert_extension_array_equal(result, expected)