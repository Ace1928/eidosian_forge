import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('type_', (tuple, list, np.array))
def test_value_vars_types(self, type_, df):
    expected = DataFrame({'id1': df['id1'].tolist() * 2, 'id2': df['id2'].tolist() * 2, 'variable': ['A'] * 10 + ['B'] * 10, 'value': df['A'].tolist() + df['B'].tolist()}, columns=['id1', 'id2', 'variable', 'value'])
    result = df.melt(id_vars=['id1', 'id2'], value_vars=type_(('A', 'B')))
    tm.assert_frame_equal(result, expected)