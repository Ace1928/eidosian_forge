from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@skip_pyarrow
@pytest.mark.parametrize('data', ['a,b,c,d\n1,2,3,4\n5,6,7,8', 'a,b,c,d\n1,2,3,4,\n5,6,7,8,'])
def test_usecols_index_col_false(all_parsers, data):
    parser = all_parsers
    usecols = ['a', 'c', 'd']
    expected = DataFrame({'a': [1, 5], 'c': [3, 7], 'd': [4, 8]})
    result = parser.read_csv(StringIO(data), usecols=usecols, index_col=False)
    tm.assert_frame_equal(result, expected)