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
def test_stack_positional_level_duplicate_column_names(future_stack):
    columns = MultiIndex.from_product([('x', 'y'), ('y', 'z')], names=['a', 'a'])
    df = DataFrame([[1, 1, 1, 1]], columns=columns)
    result = df.stack(0, future_stack=future_stack)
    new_columns = Index(['y', 'z'], name='a')
    new_index = MultiIndex.from_tuples([(0, 'x'), (0, 'y')], names=[None, 'a'])
    expected = DataFrame([[1, 1], [1, 1]], index=new_index, columns=new_columns)
    tm.assert_frame_equal(result, expected)