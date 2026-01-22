import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_date_query_with_non_date(self, engine, parser):
    n = 10
    df = DataFrame({'dates': date_range('1/1/2012', periods=n), 'nondate': np.arange(n)})
    result = df.query('dates == nondate', parser=parser, engine=engine)
    assert len(result) == 0
    result = df.query('dates != nondate', parser=parser, engine=engine)
    tm.assert_frame_equal(result, df)
    msg = 'Invalid comparison between dtype=datetime64\\[ns\\] and ndarray'
    for op in ['<', '>', '<=', '>=']:
        with pytest.raises(TypeError, match=msg):
            df.query(f'dates {op} nondate', parser=parser, engine=engine)