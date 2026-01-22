import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_two_backtick_variables_expr(self, df):
    res = df.eval('`B B` + `C C`')
    expect = df['B B'] + df['C C']
    tm.assert_series_equal(res, expect)