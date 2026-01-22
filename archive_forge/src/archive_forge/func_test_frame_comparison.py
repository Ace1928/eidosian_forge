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
@pytest.mark.parametrize('r_idx_type', lhs_index_types)
@pytest.mark.parametrize('c_idx_type', lhs_index_types)
def test_frame_comparison(self, engine, parser, r_idx_type, c_idx_type, idx_func_dict):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 10)), index=idx_func_dict[r_idx_type](10), columns=idx_func_dict[c_idx_type](10))
    res = pd.eval('df < 2', engine=engine, parser=parser)
    tm.assert_frame_equal(res, df < 2)
    df3 = DataFrame(np.random.default_rng(2).standard_normal(df.shape), index=df.index, columns=df.columns)
    res = pd.eval('df < df3', engine=engine, parser=parser)
    tm.assert_frame_equal(res, df < df3)