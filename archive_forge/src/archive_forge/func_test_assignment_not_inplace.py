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
@pytest.mark.xfail(reason='Unknown: Omitted test_ in name prior.')
def test_assignment_not_inplace(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 2)), columns=list('ab'))
    actual = df.eval('c = a + b', inplace=False)
    assert actual is not None
    expected = df.copy()
    expected['c'] = expected['a'] + expected['b']
    tm.assert_frame_equal(df, expected)