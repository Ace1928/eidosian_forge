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
@pytest.mark.parametrize('dropna', [True, False, lib.no_default])
def test_stack_empty_frame(dropna, future_stack):
    levels = [np.array([], dtype=np.int64), np.array([], dtype=np.int64)]
    expected = Series(dtype=np.float64, index=MultiIndex(levels=levels, codes=[[], []]))
    if future_stack and dropna is not lib.no_default:
        with pytest.raises(ValueError, match='dropna must be unspecified'):
            DataFrame(dtype=np.float64).stack(dropna=dropna, future_stack=future_stack)
    else:
        result = DataFrame(dtype=np.float64).stack(dropna=dropna, future_stack=future_stack)
        tm.assert_series_equal(result, expected)