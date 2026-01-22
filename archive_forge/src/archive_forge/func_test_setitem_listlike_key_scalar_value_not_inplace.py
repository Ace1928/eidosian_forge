from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
@pytest.mark.parametrize('value', [1.0, np.array([[1.0], [1.0]]), [[1.0], [1.0]]])
def test_setitem_listlike_key_scalar_value_not_inplace(self, value):
    cols = ['A', 'B']
    df = DataFrame(0, index=[0, 1], columns=cols)
    df_copy = df.copy()
    df_view = df[:]
    df[['B']] = value
    expected = DataFrame([[0, 1.0], [0, 1.0]], columns=cols)
    tm.assert_frame_equal(df_view, df_copy)
    tm.assert_frame_equal(df, expected)