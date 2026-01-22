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
@pytest.mark.parametrize('cols, values, expected', [(['C', 'D', 'D', 'a'], [1, 2, 3, 4], 4), (['D', 'C', 'D', 'a'], [1, 2, 3, 4], 4), (['C', 'B', 'B', 'a'], [1, 2, 3, 4], 4), (['C', 'B', 'a'], [1, 2, 3], 3), (['B', 'C', 'a'], [3, 2, 1], 1), (['C', 'a', 'B'], [3, 2, 1], 2)])
def test_setitem_same_column(self, cols, values, expected):
    df = DataFrame([values], columns=cols)
    df['a'] = df['a']
    result = df['a'].values[0]
    assert result == expected