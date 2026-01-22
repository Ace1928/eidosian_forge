from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.missing import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import Index
import pandas._testing as tm
@td.skip_if_no('pyarrow')
def test_slice_locs_negative_step_oob(self):
    index = Index(list('bcdxy'), dtype='string[pyarrow_numpy]')
    result = index[-10:5:1]
    tm.assert_index_equal(result, index)
    result = index[4:-10:-1]
    expected = Index(list('yxdcb'), dtype='string[pyarrow_numpy]')
    tm.assert_index_equal(result, expected)