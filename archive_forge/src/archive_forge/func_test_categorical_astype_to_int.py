import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_astype_to_int(self, any_int_dtype):
    df = DataFrame(data={'col1': pd.array([2.0, 1.0, 3.0])})
    df.col1 = df.col1.astype('category')
    df.col1 = df.col1.astype(any_int_dtype)
    expected = DataFrame({'col1': pd.array([2, 1, 3], dtype=any_int_dtype)})
    tm.assert_frame_equal(df, expected)