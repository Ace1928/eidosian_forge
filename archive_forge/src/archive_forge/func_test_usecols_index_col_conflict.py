from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index_col', ['b', 0])
@pytest.mark.parametrize('usecols', [['b', 'c'], [1, 2]])
def test_usecols_index_col_conflict(all_parsers, usecols, index_col, request):
    parser = all_parsers
    data = 'a,b,c,d\nA,a,1,one\nB,b,2,two'
    if parser.engine == 'pyarrow' and isinstance(usecols[0], int):
        with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
            parser.read_csv(StringIO(data), usecols=usecols, index_col=index_col)
        return
    expected = DataFrame({'c': [1, 2]}, index=Index(['a', 'b'], name='b'))
    result = parser.read_csv(StringIO(data), usecols=usecols, index_col=index_col)
    tm.assert_frame_equal(result, expected)