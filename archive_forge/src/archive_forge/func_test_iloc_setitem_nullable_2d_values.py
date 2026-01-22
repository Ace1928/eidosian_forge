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
def test_iloc_setitem_nullable_2d_values(self):
    df = DataFrame({'A': [1, 2, 3]}, dtype='Int64')
    orig = df.copy()
    df.loc[:] = df.values[:, ::-1]
    tm.assert_frame_equal(df, orig)
    df.loc[:] = pd.core.arrays.NumpyExtensionArray(df.values[:, ::-1])
    tm.assert_frame_equal(df, orig)
    df.iloc[:] = df.iloc[:, :].copy()
    tm.assert_frame_equal(df, orig)