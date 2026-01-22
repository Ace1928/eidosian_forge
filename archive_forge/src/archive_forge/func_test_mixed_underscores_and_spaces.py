import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_mixed_underscores_and_spaces(self, df):
    res = df.eval('A + `D_D D`')
    expect = df['A'] + df['D_D D']
    tm.assert_series_equal(res, expect)