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
@pytest.mark.parametrize('cmp', ('and', 'or'))
@pytest.mark.parametrize('lhs', (int, float))
@pytest.mark.parametrize('rhs', (int, float))
def test_bool_ops_fails_on_scalars(lhs, cmp, rhs, engine, parser):
    gen = {int: lambda: np.random.default_rng(2).integers(10), float: np.random.default_rng(2).standard_normal}
    mid = gen[lhs]()
    lhs = gen[lhs]()
    rhs = gen[rhs]()
    ex1 = f'lhs {cmp} mid {cmp} rhs'
    ex2 = f'lhs {cmp} mid and mid {cmp} rhs'
    ex3 = f'(lhs {cmp} mid) & (mid {cmp} rhs)'
    for ex in (ex1, ex2, ex3):
        msg = "cannot evaluate scalar only bool ops|'BoolOp' nodes are not"
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval(ex, engine=engine, parser=parser)