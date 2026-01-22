from __future__ import annotations
from functools import reduce
from itertools import product
import operator
import numpy as np
import pytest
from pandas.compat import PY312
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import (
from pandas.core.computation.engines import ENGINES
from pandas.core.computation.expr import (
from pandas.core.computation.expressions import (
from pandas.core.computation.ops import (
from pandas.core.computation.scope import DEFAULT_GLOBALS
def test_query_inplace(self):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    expected = df.copy()
    expected = expected[expected['a'] == 2]
    df.query('a == 2', inplace=True)
    tm.assert_frame_equal(expected, df)
    df = {}
    expected = {'a': 3}
    self.eval('a = 1 + 2', target=df, inplace=True)
    tm.assert_dict_equal(df, expected)