import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_invalid_suffixtype(self):
    df = DataFrame({'Aone': [1.0, 2.0], 'Atwo': [3.0, 4.0], 'Bone': [5.0, 6.0], 'X': ['X1', 'X2']})
    df['id'] = df.index
    exp_data = {'X': '', 'Aone': [], 'Atwo': [], 'Bone': [], 'id': [], 'year': [], 'A': [], 'B': []}
    expected = DataFrame(exp_data).astype({'year': np.int64})
    expected = expected.set_index(['id', 'year'])
    expected.index = expected.index.set_levels([0, 1], level=0)
    result = wide_to_long(df, ['A', 'B'], i='id', j='year')
    tm.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))