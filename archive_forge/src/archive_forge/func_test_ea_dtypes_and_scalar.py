import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_ea_dtypes_and_scalar(self):
    df = DataFrame([[1, 2], [3, 4]], columns=['a', 'b'], dtype='Float64')
    warning = RuntimeWarning if NUMEXPR_INSTALLED else None
    with tm.assert_produces_warning(warning):
        result = df.eval('c = b - 1')
    expected = DataFrame([[1, 2, 1], [3, 4, 3]], columns=['a', 'b', 'c'], dtype='Float64')
    tm.assert_frame_equal(result, expected)