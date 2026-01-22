from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.arm_slow
def test_detect_chained_assignment_setting_entire_column(self):
    df = random_text(100000)
    x = df.iloc[[0, 1, 2]]
    assert x._is_copy is not None
    x = df.iloc[[0, 1, 2, 4]]
    assert x._is_copy is not None
    indexer = df.letters.apply(lambda x: len(x) > 10)
    df = df.loc[indexer].copy()
    assert df._is_copy is None
    df['letters'] = df['letters'].apply(str.lower)