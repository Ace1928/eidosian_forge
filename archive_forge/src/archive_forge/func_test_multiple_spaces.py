import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_multiple_spaces(self, df):
    res = df.query('`C  C` > 5')
    expect = df[df['C  C'] > 5]
    tm.assert_frame_equal(res, expect)