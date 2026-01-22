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
def test_divide_decimal(self, box_with_array):
    box = box_with_array
    ser = Series([Decimal(10)])
    expected = Series([Decimal(5)])
    ser = tm.box_expected(ser, box)
    expected = tm.box_expected(expected, box)
    result = ser / Decimal(2)
    tm.assert_equal(result, expected)
    result = ser // Decimal(2)
    tm.assert_equal(result, expected)