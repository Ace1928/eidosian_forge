from __future__ import annotations
from collections import abc
from datetime import timedelta
from decimal import Decimal
import operator
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('idx', [RangeIndex(0, 10, 1), RangeIndex(0, 20, 2)])
@pytest.mark.parametrize('scalar', [1, 2])
def test_binops_index_scalar_pow(self, idx, scalar):
    result = pow(idx, scalar)
    expected = pow(Index(idx.to_numpy()), scalar)
    tm.assert_index_equal(result, expected, exact='equiv')