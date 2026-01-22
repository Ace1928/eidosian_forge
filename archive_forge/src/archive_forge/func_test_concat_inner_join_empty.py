import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_inner_join_empty(self):
    df_empty = DataFrame()
    df_a = DataFrame({'a': [1, 2]}, index=[0, 1], dtype='int64')
    df_expected = DataFrame({'a': []}, index=RangeIndex(0), dtype='int64')
    result = concat([df_a, df_empty], axis=1, join='inner')
    tm.assert_frame_equal(result, df_expected)
    result = concat([df_a, df_empty], axis=1, join='outer')
    tm.assert_frame_equal(result, df_a)