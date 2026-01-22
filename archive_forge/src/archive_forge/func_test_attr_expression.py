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
def test_attr_expression(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), columns=list('abc'))
    expr1 = 'df.a < df.b'
    expec1 = df.a < df.b
    expr2 = 'df.a + df.b + df.c'
    expec2 = df.a + df.b + df.c
    expr3 = 'df.a + df.b + df.c[df.b < 0]'
    expec3 = df.a + df.b + df.c[df.b < 0]
    exprs = (expr1, expr2, expr3)
    expecs = (expec1, expec2, expec3)
    for e, expec in zip(exprs, expecs):
        tm.assert_series_equal(expec, self.eval(e, local_dict={'df': df}))