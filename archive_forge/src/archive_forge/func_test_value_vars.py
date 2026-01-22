import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_value_vars(self, df):
    result3 = df.melt(id_vars=['id1', 'id2'], value_vars='A')
    assert len(result3) == 10
    result4 = df.melt(id_vars=['id1', 'id2'], value_vars=['A', 'B'])
    expected4 = DataFrame({'id1': df['id1'].tolist() * 2, 'id2': df['id2'].tolist() * 2, 'variable': ['A'] * 10 + ['B'] * 10, 'value': df['A'].tolist() + df['B'].tolist()}, columns=['id1', 'id2', 'variable', 'value'])
    tm.assert_frame_equal(result4, expected4)