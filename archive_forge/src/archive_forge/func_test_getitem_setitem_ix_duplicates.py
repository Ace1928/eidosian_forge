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
def test_getitem_setitem_ix_duplicates(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), index=['foo', 'foo', 'bar', 'baz', 'bar'])
    result = df.loc['foo']
    expected = df[:2]
    tm.assert_frame_equal(result, expected)
    result = df.loc['bar']
    expected = df.iloc[[2, 4]]
    tm.assert_frame_equal(result, expected)
    result = df.loc['baz']
    expected = df.iloc[3]
    tm.assert_series_equal(result, expected)