from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_multilevel_name_print(self, lexsorted_two_level_string_multiindex):
    index = lexsorted_two_level_string_multiindex
    ser = Series(range(len(index)), index=index, name='sth')
    expected = ['first  second', 'foo    one       0', '       two       1', '       three     2', 'bar    one       3', '       two       4', 'baz    two       5', '       three     6', 'qux    one       7', '       two       8', '       three     9', 'Name: sth, dtype: int64']
    expected = '\n'.join(expected)
    assert repr(ser) == expected