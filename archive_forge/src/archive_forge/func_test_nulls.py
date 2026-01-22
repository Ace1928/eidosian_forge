from __future__ import annotations
from datetime import datetime
import gc
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BaseMaskedArray
def test_nulls(self, index):
    if len(index) == 0:
        tm.assert_numpy_array_equal(index.isna(), np.array([], dtype=bool))
    elif isinstance(index, MultiIndex):
        idx = index.copy()
        msg = 'isna is not defined for MultiIndex'
        with pytest.raises(NotImplementedError, match=msg):
            idx.isna()
    elif not index.hasnans:
        tm.assert_numpy_array_equal(index.isna(), np.zeros(len(index), dtype=bool))
        tm.assert_numpy_array_equal(index.notna(), np.ones(len(index), dtype=bool))
    else:
        result = isna(index)
        tm.assert_numpy_array_equal(index.isna(), result)
        tm.assert_numpy_array_equal(index.notna(), ~result)