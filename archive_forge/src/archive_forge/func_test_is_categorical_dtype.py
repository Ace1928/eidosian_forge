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
def test_is_categorical_dtype():
    msg = 'is_categorical_dtype is deprecated'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert not com.is_categorical_dtype(object)
        assert not com.is_categorical_dtype([1, 2, 3])
        assert com.is_categorical_dtype(CategoricalDtype())
        assert com.is_categorical_dtype(pd.Categorical([1, 2, 3]))
        assert com.is_categorical_dtype(pd.CategoricalIndex([1, 2, 3]))