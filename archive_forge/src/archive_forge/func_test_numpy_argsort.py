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
def test_numpy_argsort(self, index):
    result = np.argsort(index)
    expected = index.argsort()
    tm.assert_numpy_array_equal(result, expected)
    result = np.argsort(index, kind='mergesort')
    expected = index.argsort(kind='mergesort')
    tm.assert_numpy_array_equal(result, expected)
    if isinstance(index, (CategoricalIndex, RangeIndex)):
        msg = "the 'axis' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.argsort(index, axis=1)
        msg = "the 'order' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.argsort(index, order=('a', 'b'))