import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_local_syntax(self, engine, parser):
    skip_if_no_pandas_parser(parser)
    df = DataFrame(np.random.default_rng(2).standard_normal((100, 10)), columns=list('abcdefghij'))
    b = 1
    expect = df[df.a < b]
    result = df.query('a < @b', engine=engine, parser=parser)
    tm.assert_frame_equal(result, expect)
    expect = df[df.a < df.b]
    result = df.query('a < b', engine=engine, parser=parser)
    tm.assert_frame_equal(result, expect)