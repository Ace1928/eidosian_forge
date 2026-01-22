import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_date_index_query_with_NaT_duplicates(self, engine, parser):
    n = 10
    df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
    df['dates1'] = date_range('1/1/2012', periods=n)
    df['dates3'] = date_range('1/1/2014', periods=n)
    df.loc[np.random.default_rng(2).random(n) > 0.5, 'dates1'] = pd.NaT
    return_value = df.set_index('dates1', inplace=True, drop=True)
    assert return_value is None
    msg = "'BoolOp' nodes are not implemented"
    with pytest.raises(NotImplementedError, match=msg):
        df.query('index < 20130101 < dates3', engine=engine, parser=parser)