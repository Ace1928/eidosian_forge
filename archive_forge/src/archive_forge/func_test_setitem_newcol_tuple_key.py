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
def test_setitem_newcol_tuple_key(self, float_frame):
    assert ('A', 'B') not in float_frame.columns
    float_frame['A', 'B'] = float_frame['A']
    assert ('A', 'B') in float_frame.columns
    result = float_frame['A', 'B']
    expected = float_frame['A']
    tm.assert_series_equal(result, expected, check_names=False)