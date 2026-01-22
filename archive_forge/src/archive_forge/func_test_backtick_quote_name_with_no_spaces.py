import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_backtick_quote_name_with_no_spaces(self, df):
    res = df.eval('A + `C_C`')
    expect = df['A'] + df['C_C']
    tm.assert_series_equal(res, expect)