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
def test_append_preserves_dtype(self, simple_index):
    index = simple_index
    N = len(index)
    result = index.append(index)
    assert result.dtype == index.dtype
    tm.assert_index_equal(result[:N], index, check_exact=True)
    tm.assert_index_equal(result[N:], index, check_exact=True)
    alt = index.take(list(range(N)) * 2)
    tm.assert_index_equal(result, alt, check_exact=True)