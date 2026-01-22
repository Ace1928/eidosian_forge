from __future__ import annotations
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.astype import astype_array
import pandas.core.dtypes.common as com
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
import pandas as pd
import pandas._testing as tm
from pandas.api.types import pandas_dtype
from pandas.arrays import SparseArray
def test_is_bool_dtype():
    assert not com.is_bool_dtype(int)
    assert not com.is_bool_dtype(str)
    assert not com.is_bool_dtype(pd.Series([1, 2]))
    assert not com.is_bool_dtype(pd.Series(['a', 'b'], dtype='category'))
    assert not com.is_bool_dtype(np.array(['a', 'b']))
    assert not com.is_bool_dtype(pd.Index(['a', 'b']))
    assert not com.is_bool_dtype('Int64')
    assert com.is_bool_dtype(bool)
    assert com.is_bool_dtype(np.bool_)
    assert com.is_bool_dtype(pd.Series([True, False], dtype='category'))
    assert com.is_bool_dtype(np.array([True, False]))
    assert com.is_bool_dtype(pd.Index([True, False]))
    assert com.is_bool_dtype(pd.BooleanDtype())
    assert com.is_bool_dtype(pd.array([True, False, None], dtype='boolean'))
    assert com.is_bool_dtype('boolean')