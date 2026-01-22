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
@pytest.mark.parametrize('arith1', sorted(set(ARITH_OPS_SYMS).difference(SPECIAL_CASE_ARITH_OPS_SYMS)))
def test_binary_arith_ops(self, arith1, lhs, rhs, engine, parser):
    ex = f'lhs {arith1} rhs'
    result = pd.eval(ex, engine=engine, parser=parser)
    expected = _eval_single_bin(lhs, arith1, rhs, engine)
    tm.assert_almost_equal(result, expected)
    ex = f'lhs {arith1} rhs {arith1} rhs'
    result = pd.eval(ex, engine=engine, parser=parser)
    nlhs = _eval_single_bin(lhs, arith1, rhs, engine)
    try:
        nlhs, ghs = nlhs.align(rhs)
    except (ValueError, TypeError, AttributeError):
        return
    else:
        if engine == 'numexpr':
            import numexpr as ne
            expected = ne.evaluate(f'nlhs {arith1} ghs')
            tm.assert_almost_equal(result.values, expected)
        else:
            expected = eval(f'nlhs {arith1} ghs')
            tm.assert_almost_equal(result, expected)