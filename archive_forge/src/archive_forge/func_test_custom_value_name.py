import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_custom_value_name(self, df, value_name):
    result10 = df.melt(value_name=value_name)
    assert result10.columns.tolist() == ['variable', 'val']
    result11 = df.melt(id_vars=['id1'], value_name=value_name)
    assert result11.columns.tolist() == ['id1', 'variable', 'val']
    result12 = df.melt(id_vars=['id1', 'id2'], value_name=value_name)
    assert result12.columns.tolist() == ['id1', 'id2', 'variable', 'val']
    result13 = df.melt(id_vars=['id1', 'id2'], value_vars='A', value_name=value_name)
    assert result13.columns.tolist() == ['id1', 'id2', 'variable', 'val']
    result14 = df.melt(id_vars=['id1', 'id2'], value_vars=['A', 'B'], value_name=value_name)
    expected14 = DataFrame({'id1': df['id1'].tolist() * 2, 'id2': df['id2'].tolist() * 2, 'variable': ['A'] * 10 + ['B'] * 10, value_name: df['A'].tolist() + df['B'].tolist()}, columns=['id1', 'id2', 'variable', value_name])
    tm.assert_frame_equal(result14, expected14)