from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
def test_isin_int_df_string_search(self):
    """Comparing df with int`s (1,2) with a string at isin() ("1")
        -> should not match values because int 1 is not equal str 1"""
    df = DataFrame({'values': [1, 2]})
    result = df.isin(['1'])
    expected_false = DataFrame({'values': [False, False]})
    tm.assert_frame_equal(result, expected_false)