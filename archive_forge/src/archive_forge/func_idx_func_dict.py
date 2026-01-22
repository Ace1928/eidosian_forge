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
@pytest.fixture
def idx_func_dict():
    return {'i': lambda n: Index(np.arange(n), dtype=np.int64), 'f': lambda n: Index(np.arange(n), dtype=np.float64), 's': lambda n: Index([f'{i}_{chr(i)}' for i in range(97, 97 + n)]), 'dt': lambda n: date_range('2020-01-01', periods=n), 'td': lambda n: timedelta_range('1 day', periods=n), 'p': lambda n: period_range('2020-01-01', periods=n, freq='D')}