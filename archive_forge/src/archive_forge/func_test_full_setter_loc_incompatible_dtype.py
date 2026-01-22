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
def test_full_setter_loc_incompatible_dtype():
    df = DataFrame({'a': [1, 2]})
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        df.loc[:, 'a'] = True
    expected = DataFrame({'a': [True, True]})
    tm.assert_frame_equal(df, expected)
    df = DataFrame({'a': [1, 2]})
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        df.loc[:, 'a'] = {0: 3.5, 1: 4.5}
    expected = DataFrame({'a': [3.5, 4.5]})
    tm.assert_frame_equal(df, expected)
    df = DataFrame({'a': [1, 2]})
    df.loc[:, 'a'] = {0: 3, 1: 4}
    expected = DataFrame({'a': [3, 4]})
    tm.assert_frame_equal(df, expected)