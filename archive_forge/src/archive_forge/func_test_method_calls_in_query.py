import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_method_calls_in_query(self, engine, parser):
    n = 10
    df = DataFrame({'a': 2 * np.random.default_rng(2).random(n), 'b': np.random.default_rng(2).random(n)})
    expected = df[df['a'].astype('int') == 0]
    result = df.query("a.astype('int') == 0", engine=engine, parser=parser)
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'a': np.where(np.random.default_rng(2).random(n) < 0.5, np.nan, np.random.default_rng(2).standard_normal(n)), 'b': np.random.default_rng(2).standard_normal(n)})
    expected = df[df['a'].notnull()]
    result = df.query('a.notnull()', engine=engine, parser=parser)
    tm.assert_frame_equal(result, expected)