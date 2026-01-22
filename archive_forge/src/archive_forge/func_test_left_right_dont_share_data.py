from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_unsigned_integer_dtype
from pandas.core.dtypes.dtypes import IntervalDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
import pandas.core.common as com
def test_left_right_dont_share_data(self):
    breaks = np.arange(5)
    result = IntervalIndex.from_breaks(breaks)._data
    assert result._left.base is None or result._left.base is not result._right.base