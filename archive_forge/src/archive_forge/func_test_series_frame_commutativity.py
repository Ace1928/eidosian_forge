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
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('c_idx_type', index_types)
@pytest.mark.parametrize('r_idx_type', lhs_index_types)
@pytest.mark.parametrize('index_name', ['index', 'columns'])
@pytest.mark.parametrize('op', ['+', '*'])
def test_series_frame_commutativity(self, engine, parser, index_name, op, r_idx_type, c_idx_type, idx_func_dict):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 10)), index=idx_func_dict[r_idx_type](10), columns=idx_func_dict[c_idx_type](10))
    index = getattr(df, index_name)
    s = Series(np.random.default_rng(2).standard_normal(5), index[:5])
    lhs = f's {op} df'
    rhs = f'df {op} s'
    if should_warn(df.index, s.index):
        with tm.assert_produces_warning(RuntimeWarning):
            a = pd.eval(lhs, engine=engine, parser=parser)
        with tm.assert_produces_warning(RuntimeWarning):
            b = pd.eval(rhs, engine=engine, parser=parser)
    else:
        a = pd.eval(lhs, engine=engine, parser=parser)
        b = pd.eval(rhs, engine=engine, parser=parser)
    if r_idx_type != 'dt' and c_idx_type != 'dt':
        if engine == 'numexpr':
            tm.assert_frame_equal(a, b)