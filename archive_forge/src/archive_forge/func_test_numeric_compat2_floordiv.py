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
@pytest.mark.parametrize('idx, div, expected', [(RangeIndex(0, 1000, 2), 2, RangeIndex(0, 500, 1)), (RangeIndex(-99, -201, -3), -3, RangeIndex(33, 67, 1)), (RangeIndex(0, 1000, 1), 2, Index(RangeIndex(0, 1000, 1)._values) // 2), (RangeIndex(0, 100, 1), 2.0, Index(RangeIndex(0, 100, 1)._values) // 2.0), (RangeIndex(0), 50, RangeIndex(0)), (RangeIndex(2, 4, 2), 3, RangeIndex(0, 1, 1)), (RangeIndex(-5, -10, -6), 4, RangeIndex(-2, -1, 1)), (RangeIndex(-100, -200, 3), 2, RangeIndex(0))])
def test_numeric_compat2_floordiv(self, idx, div, expected):
    tm.assert_index_equal(idx // div, expected, exact=True)