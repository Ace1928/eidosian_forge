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
@pytest.mark.parametrize('fill_value', [None, 0])
def test_stack_unstack_empty_frame(dropna, fill_value, future_stack):
    if future_stack and dropna is not lib.no_default:
        with pytest.raises(ValueError, match='dropna must be unspecified'):
            DataFrame(dtype=np.int64).stack(dropna=dropna, future_stack=future_stack).unstack(fill_value=fill_value)
    else:
        result = DataFrame(dtype=np.int64).stack(dropna=dropna, future_stack=future_stack).unstack(fill_value=fill_value)
        expected = DataFrame(dtype=np.int64)
        tm.assert_frame_equal(result, expected)