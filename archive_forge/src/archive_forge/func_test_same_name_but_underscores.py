import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_same_name_but_underscores(self, df):
    res = df.eval('C_C + `C C`')
    expect = df['C_C'] + df['C C']
    tm.assert_series_equal(res, expect)