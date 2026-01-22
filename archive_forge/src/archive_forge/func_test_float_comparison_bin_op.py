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
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('expr', ['x < -0.1', '-5 > x'])
def test_float_comparison_bin_op(self, dtype, expr):
    df = DataFrame({'x': np.array([0], dtype=dtype)})
    res = df.eval(expr)
    assert res.values == np.array([False])