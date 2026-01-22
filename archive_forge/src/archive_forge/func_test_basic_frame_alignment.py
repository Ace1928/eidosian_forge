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
@pytest.mark.parametrize('lr_idx_type', lhs_index_types)
@pytest.mark.parametrize('rr_idx_type', index_types)
@pytest.mark.parametrize('c_idx_type', index_types)
def test_basic_frame_alignment(self, engine, parser, lr_idx_type, rr_idx_type, c_idx_type, idx_func_dict):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 10)), index=idx_func_dict[lr_idx_type](10), columns=idx_func_dict[c_idx_type](10))
    df2 = DataFrame(np.random.default_rng(2).standard_normal((20, 10)), index=idx_func_dict[rr_idx_type](20), columns=idx_func_dict[c_idx_type](10))
    if should_warn(df.index, df2.index):
        with tm.assert_produces_warning(RuntimeWarning):
            res = pd.eval('df + df2', engine=engine, parser=parser)
    else:
        res = pd.eval('df + df2', engine=engine, parser=parser)
    tm.assert_frame_equal(res, df + df2)