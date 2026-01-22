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
@pytest.mark.parametrize('rhs', [True, False])
@pytest.mark.parametrize('lhs', [True, False])
@pytest.mark.parametrize('op', expr.BOOL_OPS_SYMS)
def test_simple_bool_ops(self, rhs, lhs, op):
    ex = f'{lhs} {op} {rhs}'
    if parser == 'python' and op in ['and', 'or']:
        msg = "'BoolOp' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            self.eval(ex)
        return
    res = self.eval(ex)
    exp = eval(ex)
    assert res == exp