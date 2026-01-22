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
def test_lhs_expression_subscript(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
    result = self.eval('(df + 1)[df > 2]', local_dict={'df': df})
    expected = (df + 1)[df > 2]
    tm.assert_frame_equal(result, expected)