from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.parametrize('idx1,idx2', [(Index(np.arange(5), dtype=np.int64), RangeIndex(5)), (Index(np.arange(5), dtype=np.float64), Index(np.arange(5), dtype=np.int64)), (Index(np.arange(5), dtype=np.float64), RangeIndex(5)), (Index(np.arange(5), dtype=np.float64), Index(np.arange(5), dtype=np.uint64))])
def test_compatible_inconsistent_pairs(idx1, idx2):
    res1 = idx1.union(idx2)
    res2 = idx2.union(idx1)
    assert res1.dtype in (idx1.dtype, idx2.dtype)
    assert res2.dtype in (idx1.dtype, idx2.dtype)