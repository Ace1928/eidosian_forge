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
def test_is_numeric_dtype():
    assert not com.is_numeric_dtype(str)
    assert not com.is_numeric_dtype(np.datetime64)
    assert not com.is_numeric_dtype(np.timedelta64)
    assert not com.is_numeric_dtype(np.array(['a', 'b']))
    assert not com.is_numeric_dtype(np.array([], dtype=np.timedelta64))
    assert com.is_numeric_dtype(int)
    assert com.is_numeric_dtype(float)
    assert com.is_numeric_dtype(np.uint64)
    assert com.is_numeric_dtype(pd.Series([1, 2]))
    assert com.is_numeric_dtype(pd.Index([1, 2.0]))

    class MyNumericDType(ExtensionDtype):

        @property
        def type(self):
            return str

        @property
        def name(self):
            raise NotImplementedError

        @classmethod
        def construct_array_type(cls):
            raise NotImplementedError

        def _is_numeric(self) -> bool:
            return True
    assert com.is_numeric_dtype(MyNumericDType())