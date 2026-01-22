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
def test_setitem_frame_upcast(self):
    df = DataFrame([[1, 2, 'foo'], [3, 4, 'bar']], columns=['A', 'B', 'C'])
    df2 = df.copy()
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        df2.loc[:, ['A', 'B']] = df.loc[:, ['A', 'B']] + 0.5
    expected = df.reindex(columns=['A', 'B'])
    expected += 0.5
    expected['C'] = df['C']
    tm.assert_frame_equal(df2, expected)