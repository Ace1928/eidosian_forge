import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_string_rank_grouping():
    df = DataFrame({'A': [1, 1, 2], 'B': [1, 2, 3]})
    result = df.groupby('A').transform('rank')
    expected = DataFrame({'B': [1.0, 2.0, 1.0]})
    tm.assert_frame_equal(result, expected)