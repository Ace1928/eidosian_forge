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
def test_equals(self, index):
    if isinstance(index, IntervalIndex):
        return
    is_ea_idx = type(index) is Index and (not isinstance(index.dtype, np.dtype))
    assert index.equals(index)
    assert index.equals(index.copy())
    if not is_ea_idx:
        assert index.equals(index.astype(object))
    assert not index.equals(list(index))
    assert not index.equals(np.array(index))
    if not isinstance(index, RangeIndex) and (not is_ea_idx):
        same_values = Index(index, dtype=object)
        assert index.equals(same_values)
        assert same_values.equals(index)
    if index.nlevels == 1:
        assert not index.equals(Series(index))