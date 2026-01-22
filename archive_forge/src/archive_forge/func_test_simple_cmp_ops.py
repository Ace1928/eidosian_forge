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
@pytest.mark.parametrize('cmp_op', expr.CMP_OPS_SYMS)
def test_simple_cmp_ops(self, cmp_op, lhs, rhs, engine, parser):
    lhs = lhs < 0
    rhs = rhs < 0
    if parser == 'python' and cmp_op in ['in', 'not in']:
        msg = "'(In|NotIn)' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            ex = f'lhs {cmp_op} rhs'
            pd.eval(ex, engine=engine, parser=parser)
        return
    ex = f'lhs {cmp_op} rhs'
    msg = '|'.join(["only list-like( or dict-like)? objects are allowed to be passed to (DataFrame\\.)?isin\\(\\), you passed a (`|')bool(`|')", "argument of type 'bool' is not iterable"])
    if cmp_op in ('in', 'not in') and (not is_list_like(rhs)):
        with pytest.raises(TypeError, match=msg):
            pd.eval(ex, engine=engine, parser=parser, local_dict={'lhs': lhs, 'rhs': rhs})
    else:
        expected = _eval_single_bin(lhs, cmp_op, rhs, engine)
        result = pd.eval(ex, engine=engine, parser=parser)
        tm.assert_equal(result, expected)