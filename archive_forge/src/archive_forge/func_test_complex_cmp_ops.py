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
@pytest.mark.parametrize('cmp1', ['!=', '==', '<=', '>=', '<', '>'], ids=['ne', 'eq', 'le', 'ge', 'lt', 'gt'])
@pytest.mark.parametrize('cmp2', ['>', '<'], ids=['gt', 'lt'])
@pytest.mark.parametrize('binop', expr.BOOL_OPS_SYMS)
def test_complex_cmp_ops(self, cmp1, cmp2, binop, lhs, rhs, engine, parser):
    if parser == 'python' and binop in ['and', 'or']:
        msg = "'BoolOp' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            ex = f'(lhs {cmp1} rhs) {binop} (lhs {cmp2} rhs)'
            pd.eval(ex, engine=engine, parser=parser)
        return
    lhs_new = _eval_single_bin(lhs, cmp1, rhs, engine)
    rhs_new = _eval_single_bin(lhs, cmp2, rhs, engine)
    expected = _eval_single_bin(lhs_new, binop, rhs_new, engine)
    ex = f'(lhs {cmp1} rhs) {binop} (lhs {cmp2} rhs)'
    result = pd.eval(ex, engine=engine, parser=parser)
    tm.assert_equal(result, expected)