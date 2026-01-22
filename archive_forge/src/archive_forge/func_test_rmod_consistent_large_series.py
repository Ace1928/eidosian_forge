from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_rmod_consistent_large_series():
    result = Series([2] * 10001).rmod(-1)
    expected = Series([1] * 10001)
    tm.assert_series_equal(result, expected)