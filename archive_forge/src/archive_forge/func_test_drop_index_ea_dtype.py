import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('idx, level', [(['a', 'b'], 0), (['a'], None)])
def test_drop_index_ea_dtype(self, any_numeric_ea_dtype, idx, level):
    df = DataFrame({'a': [1, 2, 2, pd.NA], 'b': 100}, dtype=any_numeric_ea_dtype).set_index(idx)
    result = df.drop(Index([2, pd.NA]), level=level)
    expected = DataFrame({'a': [1], 'b': 100}, dtype=any_numeric_ea_dtype).set_index(idx)
    tm.assert_frame_equal(result, expected)