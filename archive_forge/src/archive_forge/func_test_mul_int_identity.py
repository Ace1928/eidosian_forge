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
@pytest.mark.parametrize('op', [operator.mul, ops.rmul, operator.floordiv])
def test_mul_int_identity(self, op, numeric_idx, box_with_array):
    idx = numeric_idx
    idx = tm.box_expected(idx, box_with_array)
    result = op(idx, 1)
    tm.assert_equal(result, idx)