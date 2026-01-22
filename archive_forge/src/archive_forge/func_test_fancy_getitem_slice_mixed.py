from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
@td.skip_array_manager_invalid_test
def test_fancy_getitem_slice_mixed(self, float_frame, float_string_frame, using_copy_on_write, warn_copy_on_write):
    sliced = float_string_frame.iloc[:, -3:]
    assert sliced['D'].dtype == np.float64
    original = float_frame.copy()
    sliced = float_frame.iloc[:, -3:]
    assert np.shares_memory(sliced['C']._values, float_frame['C']._values)
    with tm.assert_cow_warning(warn_copy_on_write):
        sliced.loc[:, 'C'] = 4.0
    if not using_copy_on_write:
        assert (float_frame['C'] == 4).all()
        np.shares_memory(sliced['C']._values, float_frame['C']._values)
    else:
        tm.assert_frame_equal(float_frame, original)