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
def test_stack_order_with_unsorted_levels_multi_row(self, future_stack):
    mi = MultiIndex(levels=[['A', 'C', 'B'], ['B', 'A', 'C']], codes=[np.repeat(range(3), 3), np.tile(range(3), 3)])
    df = DataFrame(columns=mi, index=range(5), data=np.arange(5 * len(mi)).reshape(5, -1))
    assert all((df.loc[row, col] == df.stack(0, future_stack=future_stack).loc[(row, col[0]), col[1]] for row in df.index for col in df.columns))