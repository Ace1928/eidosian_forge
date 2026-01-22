import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_num_string_disambiguation(self):
    df = DataFrame({'A11': ['a11', 'a22', 'a33'], 'A12': ['a21', 'a22', 'a23'], 'B11': ['b11', 'b12', 'b13'], 'B12': ['b21', 'b22', 'b23'], 'BB11': [1, 2, 3], 'BB12': [4, 5, 6], 'Arating': [91, 92, 93], 'Arating_old': [91, 92, 93]})
    df['id'] = df.index
    expected = DataFrame({'Arating': [91, 92, 93, 91, 92, 93], 'Arating_old': [91, 92, 93, 91, 92, 93], 'A': ['a11', 'a22', 'a33', 'a21', 'a22', 'a23'], 'B': ['b11', 'b12', 'b13', 'b21', 'b22', 'b23'], 'BB': [1, 2, 3, 4, 5, 6], 'id': [0, 1, 2, 0, 1, 2], 'year': [11, 11, 11, 12, 12, 12]})
    expected = expected.set_index(['id', 'year'])[['Arating', 'Arating_old', 'A', 'B', 'BB']]
    result = wide_to_long(df, ['A', 'B', 'BB'], i='id', j='year')
    tm.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))