from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('df, to_replace, exp', [({'col1': [1, 2, 3], 'col2': [4, 5, 6]}, {4: 5, 5: 6, 6: 7}, {'col1': [1, 2, 3], 'col2': [5, 6, 7]}), ({'col1': [1, 2, 3], 'col2': ['4', '5', '6']}, {'4': '5', '5': '6', '6': '7'}, {'col1': [1, 2, 3], 'col2': ['5', '6', '7']})])
def test_replace_commutative(self, df, to_replace, exp):
    df = DataFrame(df)
    expected = DataFrame(exp)
    result = df.replace(to_replace)
    tm.assert_frame_equal(result, expected)