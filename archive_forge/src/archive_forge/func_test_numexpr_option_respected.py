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
@td.skip_if_no('numexpr')
@pytest.mark.parametrize(('use_numexpr', 'expected'), ((True, 'numexpr'), (False, 'python')))
def test_numexpr_option_respected(use_numexpr, expected):
    from pandas.core.computation.eval import _check_engine
    with pd.option_context('compute.use_numexpr', use_numexpr):
        result = _check_engine(None)
        assert result == expected