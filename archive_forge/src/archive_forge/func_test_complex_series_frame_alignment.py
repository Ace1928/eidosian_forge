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
@pytest.mark.filterwarnings('always::RuntimeWarning')
@pytest.mark.parametrize('r1', lhs_index_types)
@pytest.mark.parametrize('c1', index_types)
@pytest.mark.parametrize('r2', index_types)
@pytest.mark.parametrize('c2', index_types)
def test_complex_series_frame_alignment(self, engine, parser, r1, c1, r2, c2, idx_func_dict):
    n = 3
    m1 = 5
    m2 = 2 * m1
    df = DataFrame(np.random.default_rng(2).standard_normal((m1, n)), index=idx_func_dict[r1](m1), columns=idx_func_dict[c1](n))
    df2 = DataFrame(np.random.default_rng(2).standard_normal((m2, n)), index=idx_func_dict[r2](m2), columns=idx_func_dict[c2](n))
    index = df2.columns
    ser = Series(np.random.default_rng(2).standard_normal(n), index[:n])
    if r2 == 'dt' or c2 == 'dt':
        if engine == 'numexpr':
            expected2 = df2.add(ser)
        else:
            expected2 = df2 + ser
    else:
        expected2 = df2 + ser
    if r1 == 'dt' or c1 == 'dt':
        if engine == 'numexpr':
            expected = expected2.add(df)
        else:
            expected = expected2 + df
    else:
        expected = expected2 + df
    if should_warn(df2.index, ser.index, df.index):
        with tm.assert_produces_warning(RuntimeWarning):
            res = pd.eval('df2 + ser + df', engine=engine, parser=parser)
    else:
        res = pd.eval('df2 + ser + df', engine=engine, parser=parser)
    assert res.shape == expected.shape
    tm.assert_frame_equal(res, expected)