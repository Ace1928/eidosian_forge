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
def test_float_truncation(self, engine, parser):
    exp = '1000000000.006'
    result = pd.eval(exp, engine=engine, parser=parser)
    expected = np.float64(exp)
    assert result == expected
    df = DataFrame({'A': [1000000000.0009, 1000000000.0011, 1000000000.0015]})
    cutoff = 1000000000.0006
    result = df.query(f'A < {cutoff:.4f}')
    assert result.empty
    cutoff = 1000000000.001
    result = df.query(f'A > {cutoff:.4f}')
    expected = df.loc[[1, 2], :]
    tm.assert_frame_equal(expected, result)
    exact = 1000000000.0011
    result = df.query(f'A == {exact:.4f}')
    expected = df.loc[[1], :]
    tm.assert_frame_equal(expected, result)