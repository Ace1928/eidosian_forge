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
@pytest.mark.parametrize('dtype', ['int64', 'Int64'])
def test_setitem_iloc_with_numpy_array(self, dtype):
    df = DataFrame({'a': np.ones(3)}, dtype=dtype)
    df.iloc[np.array([0]), np.array([0])] = np.array([[2]])
    expected = DataFrame({'a': [2, 1, 1]}, dtype=dtype)
    tm.assert_frame_equal(df, expected)