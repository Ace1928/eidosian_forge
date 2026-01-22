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
def test_basic_period_index_boolean_expression(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((2, 2)), columns=period_range('2020-01-01', freq='D', periods=2))
    e = df < 2
    r = self.eval('df < 2', local_dict={'df': df})
    x = df < 2
    tm.assert_frame_equal(r, e)
    tm.assert_frame_equal(x, e)