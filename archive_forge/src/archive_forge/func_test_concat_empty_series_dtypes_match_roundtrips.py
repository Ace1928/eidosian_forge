import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['float64', 'int8', 'uint8', 'bool', 'm8[ns]', 'M8[ns]'])
def test_concat_empty_series_dtypes_match_roundtrips(self, dtype):
    dtype = np.dtype(dtype)
    result = concat([Series(dtype=dtype)])
    assert result.dtype == dtype
    result = concat([Series(dtype=dtype), Series(dtype=dtype)])
    assert result.dtype == dtype