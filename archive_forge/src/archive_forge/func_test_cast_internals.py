from datetime import (
import itertools
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.internals.blocks import NumpyBlock
def test_cast_internals(self, float_frame):
    msg = 'Passing a BlockManager to DataFrame'
    with tm.assert_produces_warning(DeprecationWarning, match=msg, check_stacklevel=False):
        casted = DataFrame(float_frame._mgr, dtype=int)
    expected = DataFrame(float_frame._series, dtype=int)
    tm.assert_frame_equal(casted, expected)
    with tm.assert_produces_warning(DeprecationWarning, match=msg, check_stacklevel=False):
        casted = DataFrame(float_frame._mgr, dtype=np.int32)
    expected = DataFrame(float_frame._series, dtype=np.int32)
    tm.assert_frame_equal(casted, expected)