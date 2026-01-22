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
def test_constructor_unwraps_index(self, dtype):
    index_cls = self._index_cls
    idx = Index([1, 2], dtype=dtype)
    result = index_cls(idx)
    expected = np.array([1, 2], dtype=idx.dtype)
    tm.assert_numpy_array_equal(result._data, expected)