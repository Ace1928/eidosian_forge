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
def test_modulus(self, lhs, rhs, engine, parser):
    ex = 'lhs % rhs'
    result = pd.eval(ex, engine=engine, parser=parser)
    expected = lhs % rhs
    tm.assert_almost_equal(result, expected)
    if engine == 'numexpr':
        import numexpr as ne
        expected = ne.evaluate('expected % rhs')
        if isinstance(result, (DataFrame, Series)):
            tm.assert_almost_equal(result.values, expected)
        else:
            tm.assert_almost_equal(result, expected.item())
    else:
        expected = _eval_single_bin(expected, '%', rhs, engine)
        tm.assert_almost_equal(result, expected)