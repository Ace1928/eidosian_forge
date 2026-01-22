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
@pytest.mark.parametrize('cmp1', ['<', '>'])
@pytest.mark.parametrize('cmp2', ['<', '>'])
def test_chained_cmp_op(self, cmp1, cmp2, lhs, midhs, rhs, engine, parser):
    mid = midhs
    if parser == 'python':
        ex1 = f'lhs {cmp1} mid {cmp2} rhs'
        msg = "'BoolOp' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval(ex1, engine=engine, parser=parser)
        return
    lhs_new = _eval_single_bin(lhs, cmp1, mid, engine)
    rhs_new = _eval_single_bin(mid, cmp2, rhs, engine)
    if lhs_new is not None and rhs_new is not None:
        ex1 = f'lhs {cmp1} mid {cmp2} rhs'
        ex2 = f'lhs {cmp1} mid and mid {cmp2} rhs'
        ex3 = f'(lhs {cmp1} mid) & (mid {cmp2} rhs)'
        expected = _eval_single_bin(lhs_new, '&', rhs_new, engine)
        for ex in (ex1, ex2, ex3):
            result = pd.eval(ex, engine=engine, parser=parser)
            tm.assert_almost_equal(result, expected)