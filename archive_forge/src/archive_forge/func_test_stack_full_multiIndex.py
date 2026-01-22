from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_full_multiIndex(self, future_stack):
    full_multiindex = MultiIndex.from_tuples([('B', 'x'), ('B', 'z'), ('A', 'y'), ('C', 'x'), ('C', 'u')], names=['Upper', 'Lower'])
    df = DataFrame(np.arange(6).reshape(2, 3), columns=full_multiindex[[0, 1, 3]])
    dropna = False if not future_stack else lib.no_default
    result = df.stack(dropna=dropna, future_stack=future_stack)
    expected = DataFrame([[0, 2], [1, np.nan], [3, 5], [4, np.nan]], index=MultiIndex(levels=[[0, 1], ['u', 'x', 'y', 'z']], codes=[[0, 0, 1, 1], [1, 3, 1, 3]], names=[None, 'Lower']), columns=Index(['B', 'C'], name='Upper'))
    expected['B'] = expected['B'].astype(df.dtypes.iloc[0])
    tm.assert_frame_equal(result, expected)