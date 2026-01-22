from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
def test_usecols_index_col_middle(all_parsers):
    parser = all_parsers
    data = 'a,b,c,d\n1,2,3,4\n'
    result = parser.read_csv(StringIO(data), usecols=['b', 'c', 'd'], index_col='c')
    expected = DataFrame({'b': [2], 'd': [4]}, index=Index([3], name='c'))
    tm.assert_frame_equal(result, expected)