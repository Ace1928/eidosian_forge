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
@pytest.mark.parametrize('indexer', ['B', ['B']])
def test_setitem_frame_length_0_str_key(self, indexer):
    df = DataFrame(columns=['A', 'B'])
    other = DataFrame({'B': [1, 2]})
    df[indexer] = other
    expected = DataFrame({'A': [np.nan] * 2, 'B': [1, 2]})
    expected['A'] = expected['A'].astype('object')
    tm.assert_frame_equal(df, expected)