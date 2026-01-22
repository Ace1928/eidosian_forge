import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
@pytest.mark.parametrize('op, func', [['<', operator.lt], ['>', operator.gt], ['<=', operator.le], ['>=', operator.ge]])
def test_query_lex_compare_strings(self, parser, engine, op, func, using_infer_string):
    a = Series(np.random.default_rng(2).choice(list('abcde'), 20))
    b = Series(np.arange(a.size))
    df = DataFrame({'X': a, 'Y': b})
    warning = RuntimeWarning if using_infer_string and engine == 'numexpr' else None
    with tm.assert_produces_warning(warning):
        res = df.query(f'X {op} "d"', engine=engine, parser=parser)
    expected = df[func(df.X, 'd')]
    tm.assert_frame_equal(res, expected)