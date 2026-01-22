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
def test_modulo(self, numeric_idx, box_with_array):
    box = box_with_array
    idx = numeric_idx
    expected = Index(idx.values % 2)
    idx = tm.box_expected(idx, box)
    expected = tm.box_expected(expected, box)
    result = idx % 2
    tm.assert_equal(result, expected)