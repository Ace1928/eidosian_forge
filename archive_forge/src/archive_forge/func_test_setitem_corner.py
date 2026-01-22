from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setitem_corner(self, float_frame, using_infer_string):
    df = DataFrame({'B': [1.0, 2.0, 3.0], 'C': ['a', 'b', 'c']}, index=np.arange(3))
    del df['B']
    df['B'] = [1.0, 2.0, 3.0]
    assert 'B' in df
    assert len(df.columns) == 2
    df['A'] = 'beginning'
    df['E'] = 'foo'
    df['D'] = 'bar'
    df[datetime.now()] = 'date'
    df[datetime.now()] = 5.0
    dm = DataFrame(index=float_frame.index)
    dm['A'] = 'foo'
    dm['B'] = 'bar'
    assert len(dm.columns) == 2
    assert dm.values.dtype == np.object_
    dm['C'] = 1
    assert dm['C'].dtype == np.int64
    dm['E'] = 1.0
    assert dm['E'].dtype == np.float64
    dm['A'] = 'bar'
    assert 'bar' == dm['A'].iloc[0]
    dm = DataFrame(index=np.arange(3))
    dm['A'] = 1
    dm['foo'] = 'bar'
    del dm['foo']
    dm['foo'] = 'bar'
    if using_infer_string:
        assert dm['foo'].dtype == 'string'
    else:
        assert dm['foo'].dtype == np.object_
    dm['coercible'] = ['1', '2', '3']
    if using_infer_string:
        assert dm['coercible'].dtype == 'string'
    else:
        assert dm['coercible'].dtype == np.object_