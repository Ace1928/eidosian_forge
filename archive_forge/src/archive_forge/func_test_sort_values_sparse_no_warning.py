import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('dtype', [np.uint8, bool])
def test_sort_values_sparse_no_warning(self, dtype):
    ser = pd.Series(Categorical(['a', 'b', 'a'], categories=['a', 'b', 'c']))
    df = pd.get_dummies(ser, dtype=dtype, sparse=True)
    with tm.assert_produces_warning(None):
        df.sort_values(by=df.columns.tolist())