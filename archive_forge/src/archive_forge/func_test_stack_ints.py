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
def test_stack_ints(self, future_stack):
    columns = MultiIndex.from_tuples(list(itertools.product(range(3), repeat=3)))
    df = DataFrame(np.random.default_rng(2).standard_normal((30, 27)), columns=columns)
    tm.assert_frame_equal(df.stack(level=[1, 2], future_stack=future_stack), df.stack(level=1, future_stack=future_stack).stack(level=1, future_stack=future_stack))
    tm.assert_frame_equal(df.stack(level=[-2, -1], future_stack=future_stack), df.stack(level=1, future_stack=future_stack).stack(level=1, future_stack=future_stack))
    df_named = df.copy()
    return_value = df_named.columns.set_names(range(3), inplace=True)
    assert return_value is None
    tm.assert_frame_equal(df_named.stack(level=[1, 2], future_stack=future_stack), df_named.stack(level=1, future_stack=future_stack).stack(level=1, future_stack=future_stack))