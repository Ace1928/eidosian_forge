import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_empty_series_dtypes_sparse(self):
    result = concat([Series(dtype='float64').astype('Sparse'), Series(dtype='float64').astype('Sparse')])
    assert result.dtype == 'Sparse[float64]'
    result = concat([Series(dtype='float64').astype('Sparse'), Series(dtype='float64')])
    expected = pd.SparseDtype(np.float64)
    assert result.dtype == expected
    result = concat([Series(dtype='float64').astype('Sparse'), Series(dtype='object')])
    expected = pd.SparseDtype('object')
    assert result.dtype == expected