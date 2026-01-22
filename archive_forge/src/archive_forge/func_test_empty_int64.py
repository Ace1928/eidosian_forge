from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('name, expected', [['foo', 'Series([], Name: foo, dtype: int64)'], [None, 'Series([], dtype: int64)']])
def test_empty_int64(self, name, expected):
    s = Series([], dtype=np.int64, name=name)
    assert repr(s) == expected