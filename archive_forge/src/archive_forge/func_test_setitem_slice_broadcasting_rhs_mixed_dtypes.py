from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
@pytest.mark.parametrize('indexer', [tm.setitem, tm.iloc])
@pytest.mark.parametrize('box', [Series, np.array, list, pd.array])
@pytest.mark.parametrize('n', [1, 2, 3])
def test_setitem_slice_broadcasting_rhs_mixed_dtypes(self, n, box, indexer):
    df = DataFrame([[1, 3, 5], ['x', 'y', 'z']] + [[2, 4, 6]] * n, columns=['a', 'b', 'c'])
    indexer(df)[1:] = box([10, 11, 12])
    expected = DataFrame([[1, 3, 5]] + [[10, 11, 12]] * (n + 1), columns=['a', 'b', 'c'], dtype='object')
    tm.assert_frame_equal(df, expected)