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
@pytest.mark.parametrize('ex', ('1 or 2', '1 and 2', 'a and b', 'a or b', '1 or 2 and (3 + 2) > 3', '2 * x > 2 or 1 and 2', '2 * df > 3 and 1 or a'))
def test_disallow_scalar_bool_ops(self, ex, engine, parser):
    x, a, b = (np.random.default_rng(2).standard_normal(3), 1, 2)
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)))
    msg = "cannot evaluate scalar only bool ops|'BoolOp' nodes are not"
    with pytest.raises(NotImplementedError, match=msg):
        pd.eval(ex, engine=engine, parser=parser)