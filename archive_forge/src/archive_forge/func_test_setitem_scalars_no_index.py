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
def test_setitem_scalars_no_index(self):
    df = DataFrame()
    df['foo'] = 1
    expected = DataFrame(columns=['foo']).astype(np.int64)
    tm.assert_frame_equal(df, expected)