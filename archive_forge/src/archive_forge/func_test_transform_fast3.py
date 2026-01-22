import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_fast3():
    df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=['g', 'a', 'a'])
    result = df.groupby('g').transform('first')
    expected = df.drop('g', axis=1)
    tm.assert_frame_equal(result, expected)