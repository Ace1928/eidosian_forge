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
def test_setitem_slice_position(self):
    df = DataFrame(np.zeros((100, 1)))
    df[-4:] = 1
    arr = np.zeros((100, 1))
    arr[-4:] = 1
    expected = DataFrame(arr)
    tm.assert_frame_equal(df, expected)