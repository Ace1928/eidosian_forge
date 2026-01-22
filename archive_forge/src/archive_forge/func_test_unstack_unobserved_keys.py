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
def test_unstack_unobserved_keys(self, future_stack):
    levels = [[0, 1], [0, 1, 2, 3]]
    codes = [[0, 0, 1, 1], [0, 2, 0, 2]]
    index = MultiIndex(levels, codes)
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 2)), index=index)
    result = df.unstack()
    assert len(result.columns) == 4
    recons = result.stack(future_stack=future_stack)
    tm.assert_frame_equal(recons, df)