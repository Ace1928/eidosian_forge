import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_custom_var_name(self, df, var_name):
    result5 = df.melt(var_name=var_name)
    assert result5.columns.tolist() == ['var', 'value']
    result6 = df.melt(id_vars=['id1'], var_name=var_name)
    assert result6.columns.tolist() == ['id1', 'var', 'value']
    result7 = df.melt(id_vars=['id1', 'id2'], var_name=var_name)
    assert result7.columns.tolist() == ['id1', 'id2', 'var', 'value']
    result8 = df.melt(id_vars=['id1', 'id2'], value_vars='A', var_name=var_name)
    assert result8.columns.tolist() == ['id1', 'id2', 'var', 'value']
    result9 = df.melt(id_vars=['id1', 'id2'], value_vars=['A', 'B'], var_name=var_name)
    expected9 = DataFrame({'id1': df['id1'].tolist() * 2, 'id2': df['id2'].tolist() * 2, var_name: ['A'] * 10 + ['B'] * 10, 'value': df['A'].tolist() + df['B'].tolist()}, columns=['id1', 'id2', var_name, 'value'])
    tm.assert_frame_equal(result9, expected9)