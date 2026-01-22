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
def test_unstack_odd_failure(self, future_stack):
    mi = MultiIndex.from_arrays([['Fri'] * 4 + ['Sat'] * 2 + ['Sun'] * 2 + ['Thu'] * 3, ['Dinner'] * 2 + ['Lunch'] * 2 + ['Dinner'] * 5 + ['Lunch'] * 2, ['No', 'Yes'] * 4 + ['No', 'No', 'Yes']], names=['day', 'time', 'smoker'])
    df = DataFrame({'sum': np.arange(11, dtype='float64'), 'len': np.arange(11, dtype='float64')}, index=mi)
    result = df.unstack(2)
    recons = result.stack(future_stack=future_stack)
    if future_stack:
        recons = recons.dropna(how='all')
    tm.assert_frame_equal(recons, df)