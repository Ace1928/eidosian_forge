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
def test_delete_base(self, index):
    if not len(index):
        return
    if isinstance(index, RangeIndex):
        return
    expected = index[1:]
    result = index.delete(0)
    assert result.equals(expected)
    assert result.name == expected.name
    expected = index[:-1]
    result = index.delete(-1)
    assert result.equals(expected)
    assert result.name == expected.name
    length = len(index)
    msg = f'index {length} is out of bounds for axis 0 with size {length}'
    with pytest.raises(IndexError, match=msg):
        index.delete(length)