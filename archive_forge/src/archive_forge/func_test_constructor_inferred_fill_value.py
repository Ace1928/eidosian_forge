import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('data, fill_value', [(np.array([1, 2]), 0), (np.array([1.0, 2.0]), np.nan), ([True, False], False), ([pd.Timestamp('2017-01-01')], pd.NaT)])
def test_constructor_inferred_fill_value(self, data, fill_value):
    result = SparseArray(data).fill_value
    if isna(fill_value):
        assert isna(result)
    else:
        assert result == fill_value