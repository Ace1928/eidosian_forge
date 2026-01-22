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
@pytest.mark.skipif(not NUMEXPR_INSTALLED, reason='Unary ops only implemented for numexpr')
@pytest.mark.parametrize('fn', _unary_math_ops)
def test_unary_functions(self, fn):
    df = DataFrame({'a': np.random.default_rng(2).standard_normal(10)})
    a = df.a
    expr = f'{fn}(a)'
    got = self.eval(expr)
    with np.errstate(all='ignore'):
        expect = getattr(np, fn)(a)
    tm.assert_series_equal(got, expect, check_names=False)