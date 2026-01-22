from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('names,usecols', [(['b', 'c'], [1, 2]), (['a', 'b', 'c'], ['b', 'c'])])
def test_usecols_relative_to_names(all_parsers, names, usecols):
    data = '1,2,3\n4,5,6\n7,8,9\n10,11,12'
    parser = all_parsers
    if parser.engine == 'pyarrow' and (not isinstance(usecols[0], int)):
        pytest.skip(reason='https://github.com/apache/arrow/issues/38676')
    result = parser.read_csv(StringIO(data), names=names, header=None, usecols=usecols)
    expected = DataFrame([[2, 3], [5, 6], [8, 9], [11, 12]], columns=['b', 'c'])
    tm.assert_frame_equal(result, expected)