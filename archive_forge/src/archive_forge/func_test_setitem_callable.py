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
def test_setitem_callable(self):
    df = DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
    df[lambda x: 'A'] = [11, 12, 13, 14]
    exp = DataFrame({'A': [11, 12, 13, 14], 'B': [5, 6, 7, 8]})
    tm.assert_frame_equal(df, exp)