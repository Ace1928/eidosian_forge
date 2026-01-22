import re
import warnings
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
@pytest.mark.parametrize('dtype, fill_value', [('int', 0), ('float', np.nan), ('bool', False), ('object', np.nan), ('datetime64[ns]', np.datetime64('NaT', 'ns')), ('timedelta64[ns]', np.timedelta64('NaT', 'ns'))])
def test_inferred_dtype(dtype, fill_value):
    sparse_dtype = SparseDtype(dtype)
    result = sparse_dtype.fill_value
    if pd.isna(fill_value):
        assert pd.isna(result) and type(result) == type(fill_value)
    else:
        assert result == fill_value